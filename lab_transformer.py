# %% [markdown]
# # Фінальна лабораторна: збираємо T5 encoder–decoder
#
# Мета — зібрати повний Transformer із компонентів попередніх робіт,
# завантажити ваги `google-t5/t5-small` і виконати переклад власним forward.
# Заповнюйте `...`, запускайте перевірки та пояснюйте форми тензорів.
# Навчати модель не потрібно. Працюємо на CPU у float32 без Dropout і KV-cache.
#
# ## 1. Від попередніх функцій до повної мережі
#
# | Попередня робота | Що переносимо сюди |
# |---|---|
# | lab-tensors | множення матриць → `x @ weight.T` із batch-виміром |
# | lab-tokenizer | текст ↔ id; для готових ваг потрібен саме токенізатор T5 |
# | lab-attention | softmax, Q/K/V, голови, ReLU та FFN |
# | lab-positional-encoding | embedding lookup; позиції T5 задає інакше |
# | Lab-encoder | RMSNorm і pre-norm residual-гілки |
#
# Відкрийте свої розв'язки поруч. Перенесіть їхню логіку в тензорні функції
# нижче: повільні цикли над списками замінюємо операціями PyTorch, зберігаючи
# математичні дії. Не імпортуємо студентські файли: їхні демонстрації можуть
# виконуватись під час імпорту. Виклики готових T5-блоків у власному forward
# не використовуємо; Transformers потрібен для токенізації й незалежної перевірки.
#
# T5-small: приблизно 60 млн параметрів, 6 encoder + 6 decoder блоків,
# `d_model=512`, `d_ff=2048`, 8 голів по 64 координати.
# Ваги float32 займають приблизно 240 MB; разом із перевірочною моделлю
# та проміжними тензорами потрібно більше RAM. Перший запуск потребує інтернету.
#
# Встановлення в середовище Jupyter: `%pip install -r requirements.txt`.
# Для запуску Python: `python3 -m pip install -r requirements.txt`.

# %%
import json
import math
import platform
from statistics import median
from time import perf_counter
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

torch.set_num_threads(2)
MODEL_ID = "google-t5/t5-small"
# Фіксований snapshot: конфігурація, токенізатор і ваги мають одну версію.
REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
CACHE_DIR = ".cache/huggingface"

# %% [markdown]
# ## 2. Повторюємо базові компоненти
#
# Форми: `ids=(B,L)`, `x=(B,L,D)`, вага Linear — `(out,in)`.
# Перенесіть embedding lookup, множення матриць, residual, ReLU і RMSNorm.
# RMSNorm нормує **останню** вісь: `x / sqrt(mean(x²)+eps) * gamma`.
# У T5 немає віднімання середнього чи beta; epsilon беремо з config.
# Dropout у режимі inference є тотожним перетворенням.

# %%
def embedding_lookup(ids, table):
    """Виберіть рядки таблиці для всіх id."""
    ...

# %%
def linear(x, weight):
    """Лінійна проєкція без bias."""
    ...

# %%
def residual_add(x, branch):
    """Залишковий зв’язок."""
    ...

# %%
def rms_norm(x, gamma, eps):
    """RMSNorm по останньому виміру."""
    ...

# %%
def ffn(x, wi, wo):
    """FFN оригінального T5: Linear → ReLU → Linear."""
    ...

# %%
def test_primitives():
    x = torch.tensor([[[1., 3.], [0., 0.]]])
    torch.testing.assert_close(embedding_lookup(torch.tensor([[1, 0]]), x[0]), x[:, [1, 0]])
    torch.testing.assert_close(linear(x, torch.eye(2)), x)
    torch.testing.assert_close(residual_add(x, -x), torch.zeros_like(x))
    expected = x / torch.sqrt(torch.tensor([[[5.], [0.]]]) + 1e-6)
    torch.testing.assert_close(rms_norm(x, torch.ones(2), 1e-6), expected)
    torch.testing.assert_close(ffn(-x, torch.eye(2), torch.eye(2)), torch.zeros_like(x))

if __name__ == "__main__":
    test_primitives()
    print("✓ базові компоненти")

# %% [markdown]
# ## 3. Маски та позиції T5
#
# Encoder бачить усі непорожні source-токени. Decoder self-attention бачить
# лише поточний і попередні target-токени. Cross-attention: Q із decoder,
# K/V із фінального encoder; маскуємо padding джерела.
# `True` у нашій масці означає дозволений ключ. Форма `(B,1,Q,K)`
# транслюється на всі голови. Не визначайте decoder-маску через `ids != 0`:
# його стартовий id дорівнює pad id, але це справжня стартова позиція!
# У цій роботі decoder-префікси не мають padding.
#
# T5 не додає sinusoidal embedding і не використовує RoPE. До attention
# scores додається навчений relative position bias `(1,H,Q,K)`. Відстані
# групуються в buckets: близькі — точно, далекі — логарифмічно.
# Таблиця є лише в першому self-attention блоці кожного стеку; отриманий bias
# повторно використовують усі його блоки. Cross-attention не має position bias.
# Bucket-функція нижче надана: нова вправа — правильно підключити її до attention.

# %%
def attention_mask(key_mask, query_length, causal=False):
    """key_mask=(B,K), поверніть bool (B,1,Q,K)."""
    ...

# %%
def relative_position_bucket(relative_position, bidirectional, num_buckets=32, max_distance=128):
    bucket = torch.zeros_like(relative_position)
    if bidirectional:
        num_buckets //= 2
        bucket += (relative_position > 0).long() * num_buckets
        distance = relative_position.abs()
    else:
        distance = -torch.minimum(relative_position, torch.zeros_like(relative_position))
    exact = num_buckets // 2
    large = exact + (torch.log(distance.float().clamp(min=1) / exact)
                     / math.log(max_distance / exact) * (num_buckets - exact)).long()
    return bucket + torch.where(distance < exact, distance, large.clamp(max=num_buckets - 1))


def position_bias(table, q_length, k_length, bidirectional, max_distance=128):
    q = torch.arange(q_length, device=table.device)[:, None]
    k = torch.arange(k_length, device=table.device)[None, :]
    buckets = relative_position_bucket(k - q, bidirectional, table.shape[0], max_distance)
    return table[buckets].permute(2, 0, 1).unsqueeze(0)


def test_masks():
    mask = attention_mask(torch.tensor([[1, 1, 0]]), 3, True)
    assert mask[0, 0].tolist() == [[True, False, False], [True, True, False], [True, True, False]]
    distances = torch.tensor([0, -1, -2, -10000, 1])
    assert relative_position_bucket(distances, False).tolist() == [0, 1, 2, 31, 0]
    assert relative_position_bucket(torch.tensor([-1, 0, 1]), True).tolist() == [1, 0, 17]

if __name__ == "__main__":
    test_masks()
    print("✓ маски та buckets")

# %% [markdown]
# ## 4. Multi-head attention
#
# Перенесіть логіку зі своєї `multi_head_attention`.
# Проєкції `(B,L,H*d_kv)` розбийте на `(B,H,L,d_kv)`.
# Обчисліть `softmax(Q @ Kᵀ + bias + mask) @ V`, з'єднайте голови й застосуйте O.
# **У T5 немає ділення scores на sqrt(d_kv)**: масштаб уже враховано під час
# навчання ваг. Усі Q/K/V/O проєкції без bias. Softmax — по останній осі.
# Для cross-attention довжини Q та K можуть відрізнятися.

# %%
def multi_head_attention(x, context, weights, prefix, heads, mask, bias=None):
    """Self-attention при context=x; cross-attention при context=encoder_output."""
    ...

# %%
def test_attention():
    weights = {name + ".weight": torch.eye(2) for name in ("q", "k", "v", "o")}
    x = torch.tensor([[[1., 0.], [0., 1.]]])
    mask = attention_mask(torch.ones(1, 2), 2, True)
    out = multi_head_attention(x, x, weights, "", 1, mask)
    torch.testing.assert_close(out[:, 0], x[:, 0])
    expected = torch.softmax(torch.tensor([0., 1.]), 0)
    torch.testing.assert_close(out[0, 1], expected)
    context = torch.tensor([[[2., 3.], [99., 99.], [-99., -99.]]])
    mask = attention_mask(torch.tensor([[1, 0, 0]]), 2)
    out = multi_head_attention(x, context, weights, "", 1, mask)
    torch.testing.assert_close(out, context[:, :1].expand_as(x))

if __name__ == "__main__":
    test_attention()
    print("✓ self/cross attention")

# %% [markdown]
# ## 5. Складаємо encoder та decoder
#
# Encoder-блок має дві pre-norm гілки: self-attention, FFN.
# Decoder-блок має три: causal self-attention, cross-attention, FFN.
# Кожна гілка має **власну** RMSNorm і додається до поточного стану.
# Після останнього блока стеку потрібна ще фінальна RMSNorm.
#
# Імена ваг збігаються з checkpoint: `encoder.block.0.layer.0.SelfAttention.q.weight`.
# У decoder FFN має індекс layer 2, в encoder — 1.
# Напишіть спільний цикл, що викликає вже реалізовані функції.

# %%
def transformer_stack(ids, key_mask, weights, config, decoder=False, memory=None, source_mask=None):
    """Поверніть фінальні стани (B,L,D) encoder або decoder."""
    ...

# %% [markdown]
# ## 6. Від decoder до словника
#
# Encoder та decoder використовують одну embedding-таблицю; LM head також
# ділить із нею ваги. Для tied embeddings T5 масштабує фінальний decoder
# стан на `d_model**-0.5` **перед** проєкцією у словник.
# Поверніть logits `(B,target_length,vocab_size)`, без softmax.

# %%
def decode_logits(decoder_ids, memory, source_mask, weights, config):
    """Власний decoder та LM head."""
    ...

# %%
def forward(input_ids, source_mask, decoder_ids, weights, config):
    memory = transformer_stack(input_ids, source_mask, weights, config)
    return decode_logits(decoder_ids, memory, source_mask, weights, config)

# %% [markdown]
# ## 7. Спочатку перевірка на маленькій моделі без завантаження
#
# Випадкові ваги не генерують осмислений текст, зате дозволяють перевірити
# всю архітектуру швидко та без мережі. Порівнюємо logits із незалежною
# реалізацією Transformers у `eval()`. Перевіряємо batch, padding джерела,
# різні довжини та заборону підглядання у майбутнє decoder.

# %%
@torch.inference_mode()
def test_tiny_model():
    torch.manual_seed(7)
    config = T5Config(vocab_size=37, d_model=16, d_kv=4, d_ff=32,
                      num_layers=2, num_decoder_layers=2, num_heads=4,
                      dropout_rate=0., decoder_start_token_id=0, eos_token_id=1,
                      pad_token_id=0, feed_forward_proj="relu", tie_word_embeddings=True)
    reference = T5ForConditionalGeneration(config).eval()
    weights = reference.state_dict()
    source = torch.tensor([[4, 5, 1, 0], [8, 9, 10, 1]])
    mask = source.ne(0).long()
    target = torch.tensor([[0, 6, 7], [0, 11, 12]])
    actual = forward(source, mask, target, weights, config)
    expected = reference(input_ids=source, attention_mask=mask,
                         decoder_input_ids=target, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    changed = target.clone()
    changed[:, -1] = 20
    torch.testing.assert_close(actual[:, :-1], forward(source, mask, changed, weights, config)[:, :-1])
    changed_source = source.clone()
    changed_source[0, -1] = 25
    torch.testing.assert_close(actual, forward(changed_source, mask, target, weights, config), atol=2e-5, rtol=2e-5)
    print("✓ tiny T5: logits, causal mask, source padding")

if __name__ == "__main__":
    test_tiny_model()

# %% [markdown]
# ## 8. Завантажуємо справжні ваги Hugging Face
#
# Завантажуємо `config.json`, `model.safetensors` і токенізатор із одного snapshot.
# Читаємо тензори напряму; не викликаємо готову модель для нашого inference.
# Таблиці tied embeddings можуть бути збережені лише один раз — достатньо
# `shared.weight`. Для цієї роботи підтримуємо саме оригінальний T5-small
# із ReLU, без gated FFN. Не підміняйте його FLAN-T5 без адаптації.

# %%
def load_checkpoint():
    options = dict(repo_id=MODEL_ID, revision=REVISION, cache_dir=CACHE_DIR)
    config_path = hf_hub_download(filename="config.json", **options)
    config = T5Config.from_json_file(config_path)
    assert config.feed_forward_proj == "relu" and config.tie_word_embeddings
    weights = load_file(hf_hub_download(filename="model.safetensors", **options), device="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=CACHE_DIR)
    assert weights["shared.weight"].shape == (config.vocab_size, config.d_model)
    return config, weights, tokenizer

if __name__ == "__main__":
    config, weights, tokenizer = load_checkpoint()
    print("Тензорів у checkpoint:", len(weights))
    print("Embedding:", tuple(weights["shared.weight"].shape))

# %% [markdown]
# ## 9. Авторегресивна генерація
#
# Encoder рахуємо один раз. Почніть decoder із `decoder_start_token_id=0`.
# На кожному кроці обчисліть logits усього префікса, виберіть argmax останньої
# позиції та додайте id. Зупиніться на EOS=1 або після `max_new_tokens`.
# Стартовий token не входить у цей ліміт. Генеруємо один текст за раз.
# KV-cache тут не реалізуємо: повторне обчислення decoder простіше перевірити.

# %%
def greedy_generate(input_ids, source_mask, weights, config, max_new_tokens=32):
    """Поверніть id разом зі стартовим токеном; batch=1."""
    ...

# %%
if __name__ == "__main__":
    prompt = "translate English to German: The house is wonderful."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        generated = greedy_generate(inputs.input_ids, inputs.attention_mask, weights, config)
    print("Вхід:", prompt)
    print("Вихід:", tokenizer.decode(generated[0], skip_special_tokens=True))

# %% [markdown]
# ## 10. Порівнюємо з pretrained T5
#
# Тепер створюємо готову модель **лише як еталон**. Однакові ваги, CPU,
# float32, `eval()`, greedy без beam search. Спочатку порівнюємо logits для
# фіксованого decoder-префікса, потім згенеровані id. Збіг тексту сам по собі
# не доводить правильність обчислень. Невеликі похибки float32 допустимі.

# %%
@torch.inference_mode()
def compare_pretrained(config, weights, tokenizer):
    reference = T5ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISION, cache_dir=CACHE_DIR, use_safetensors=True).eval()
    for prompt in ("translate English to German: The house is wonderful.",
                   "translate English to French: I like reading books."):
        inputs = tokenizer(prompt, return_tensors="pt")
        prefix = torch.tensor([[config.decoder_start_token_id, 8, 3]])
        actual = forward(inputs.input_ids, inputs.attention_mask, prefix, weights, config)
        expected = reference(**inputs, decoder_input_ids=prefix, use_cache=False).logits
        error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        ours = greedy_generate(inputs.input_ids, inputs.attention_mask, weights, config)
        theirs = reference.generate(**inputs, max_new_tokens=32, do_sample=False,
                                    num_beams=1, use_cache=False)
        assert torch.equal(ours, theirs), (ours, theirs)
        print(f"✓ max |Δlogits|={error:.2e}; {tokenizer.decode(ours[0], skip_special_tokens=True)}")

    return reference

if __name__ == "__main__":
    reference = compare_pretrained(config, weights, tokenizer)

# %% [markdown]
# ## 11. Порівняння швидкості inference
#
# Порівняємо повну генерацію одного перекладу: encoder і всі кроки decoder.
# Власна реалізація працює без KV-cache. Для Transformers виміряємо два режими:
# `use_cache=False` для однакового обсягу повторних обчислень та `use_cache=True`,
# щоб оцінити ефект кешування. Обидва використовують greedy decoding.
#
# Працюємо на CPU, float32, batch=1, з однаковою кількістю потоків і лімітом
# токенів. Токенізація, завантаження ваг, перевірки та декодування тексту
# не входять у час. Кожен виклик починається з нового encoder проходу й
# порожнього decoder-кешу. Спочатку прогріваємо кожен режим, далі чергуємо
# порядок вимірювань, щоб зменшити вплив порядку запусків.
#
# Звітуємо медіану часу, діапазон, кількість нових токенів (разом із EOS,
# без стартового токена) та tokens/s = кількість / медіана часу.
# Це пропускна здатність усієї генерації, а не швидкість одного decoder-кроку.
# EOS може завершити генерацію раніше ліміту; перевіряємо однакові id усіх режимів.
# Відношення `час власної / час Transformers` понад 1 означає, що Transformers
# швидший. Конкретні числа залежать від CPU, довжин і версій бібліотек;
# на коротких текстах накладні витрати можуть переважати виграш від KV-cache.

# %%
@torch.inference_mode()
def benchmark_inference(config, weights, tokenizer, reference, repeats=5, warmup=2,
                        max_new_tokens=32,
                        prompt="translate English to German: The house is wonderful."):
    if repeats < 1 or warmup < 1 or max_new_tokens < 1:
        raise ValueError("repeats, warmup та max_new_tokens мають бути додатними")
    reference.eval()
    assert next(reference.parameters()).device.type == "cpu"
    assert next(reference.parameters()).dtype == torch.float32
    assert weights["shared.weight"].device.type == "cpu"
    assert weights["shared.weight"].dtype == torch.float32
    inputs = tokenizer(prompt, return_tensors="pt")
    runs = {
        "Власна, без cache": lambda: greedy_generate(
            inputs.input_ids, inputs.attention_mask, weights, config, max_new_tokens),
        "Transformers, без cache": lambda: reference.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=False),
        "Transformers, KV-cache": lambda: reference.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True),
    }
    expected = None
    for run in runs.values():
        for _ in range(warmup):
            output = run()
            if expected is None:
                expected = output.clone()
            assert torch.equal(output, expected), "Різні id: спочатку перевірте правильність inference"
    times = {name: [] for name in runs}
    names = list(runs)
    for repeat in range(repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            start = perf_counter()
            output = runs[name]()
            elapsed = perf_counter() - start
            assert torch.equal(output, expected)
            times[name].append(elapsed)
    count = expected.shape[1] - 1
    import transformers
    print(f"CPU: {platform.machine()}; torch={torch.__version__}; transformers={transformers.__version__}")
    print(f"Потоків: {torch.get_num_threads()}; source tokens={inputs.input_ids.shape[1]}; "
          f"нових токенів={count}; warmup={warmup}; repeats={repeats}")
    print(f"{'Режим':<28} {'Медіана, ms':>12} {'min–max, ms':>20} {'tokens/s':>10} {'Відношення':>12}")
    baseline = median(times[names[0]])
    rows = []
    for name, samples in times.items():
        seconds = median(samples)
        ratio = baseline / seconds
        rows.append(dict(mode=name, median_seconds=seconds, min_seconds=min(samples),
                         max_seconds=max(samples), new_tokens=count,
                         tokens_per_second=count / seconds, relative_speed=ratio))
        span = f"{min(samples)*1000:.1f}–{max(samples)*1000:.1f}"
        print(f"{name:<28} {seconds*1000:>12.1f} {span:>20} {count/seconds:>10.1f} {ratio:>11.2f}x")
    return rows

if __name__ == "__main__":
    timing_results = benchmark_inference(config, weights, tokenizer, reference)

# %% [markdown]
# **Дослідження:** повторіть вимірювання для довшого source-тексту і лімітів
# 16, 32 та 64. Запишіть фактичну довжину виходу: якщо EOS з'являється раніше,
# збільшення ліміту не збільшує обсяг роботи. Для дослідження довгих виходів
# використайте довший текст із завданням `summarize:`. Поясніть вплив KV-cache
# і причини, чому однакова математика не гарантує однаковий час виконання.

# %% [markdown]
# ## 12. Необов'язково: pretrained Mixture-of-Experts — Switch-base-8
#
# T5-small не містить MoE. Тепер завантажимо **іншу навчену модель**:
# `google/switch-base-8`. Це encoder–decoder з T5-подібними attention,
# RMSNorm, relative position bias і FFN-експертами. Можна повторно використати
# вже реалізовані функції; новий компонент — router перед експертами.
# У кожному стеку 12 блоків, кожен другий має sparse FFN із 8 експертів.
# Router обирає **одного експерта для кожного токена**, а не для всього речення.
#
# Ваги safetensors: приблизно **1.24 GB у bfloat16**, близько 619 млн параметрів.
# Для цієї CPU-вправи переводимо їх у float32: близько 2.48 GB на одну копію.
# Порівняння з еталоном потребує двох копій плюс проміжна пам'ять.
# Файл завантажується лише при запуску необов'язкової частини й кешується.
#
# Це pretrained **span-corruption** модель: вона відновлює пропущені фрагменти
# англійського тексту, позначені `<extra_id_0>`. Вона не instruction/chat модель
# і не готовий перекладач. Приклад: `The capital of France is <extra_id_0>.`
# Decoder повертає sentinel-маркери та заповнення; покажемо також відновлений текст.
# [Model card](https://huggingface.co/google/switch-base-8).
#
# ![Рисунок 2 зі статті Switch Transformers: encoder-блок і маршрутизація токенів](figures/switch_transformer_paper.svg)
#
# **Оригінальний рисунок 2:** William Fedus, Barret Zoph, Noam Shazeer,
# *Switch Transformers: Scaling to Trillion Parameter Models with Simple and
# Efficient Sparsity*, JMLR 23(120), 2022.
# [Стаття та підпис рисунка](https://arxiv.org/html/2101.03961v3#S2.F2).
# Локальна копія оригінального SVG, без перемальовування.
#
# На рисунку та у вправі однаковий **Switch top-1**: вихід обраного експерта
# множимо на його початкову softmax-ймовірність. **Не** перенормовуємо її до 1.
# Attention, RMSNorm, residual та проєкція у словник вже знайомі.
# Новий навчуваний шар — `Linear(D, E)` без bias: router.
#
# ### 12.1. Router: спочатку конкретний приклад
#
# Router — звичайний лінійний шар: кожен рядок `router_weight` відповідає
# одному експерту. Він отримує поточний стан токена **після RMSNorm**.
# Приклад із двома координатами та трьома експертами:
#
# ```text
# z = [2, 0]                 W_router = [[ 1,  0],   ← експерт 0
#                                        [ 0,  1],   ← експерт 1
#                                        [-1, -1]]   ← експерт 2
# scores = z @ W_router.T = [2, 0, -2]
# probabilities = softmax(scores) ≈ [0.8668, 0.1173, 0.0159]
# route = 0; gate ≈ 0.8668
# MoE-гілка для цього токена = 0.8668 * FFN_0(z)
# ```
#
# Для `z=[0,2]` scores дорівнюють `[0,2,-2]`: обирається експерт 1.
# Отже, одна й та сама матриця router може спрямувати різні токени до різних
# експертів. Обчислення всередині експертів уже реалізовані функцією `ffn`.
# Gate — початкова softmax-ймовірність обраного експерта, **не** одиниця.
# Номер експерта і gate мають різні ролі: перший обирає FFN, другий масштабує вихід.

# %%
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration

SWITCH_MODEL_ID = "google/switch-base-8"
# Фіксований safetensors-конвертований snapshot офіційного checkpoint (PR #5).
SWITCH_REVISION = "4f9b6f805db423183c0e7c3cda7611e43364acac"

# Готовий числовий приклад — запускається до нової вправи.
example_z = torch.tensor([[[2., 0.], [0., 2.]]])
example_router_weight = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])
example_scores = example_z @ example_router_weight.T
example_probabilities = torch.softmax(example_scores, dim=-1)
if __name__ == "__main__":
    print("Router scores:", example_scores)
    print("Router probabilities:", example_probabilities)
    print("Обрані експерти:", example_probabilities.argmax(dim=-1))

# %% [markdown]
# **Єдина нова вправа: реалізуйте лише `switch_router`.**
#
# Вхід: `z=(B,L,D)` і `router_weight=(E,D)`.
#
# 1. Обчисліть scores через раніше реалізовану `linear`.
# 2. Застосуйте softmax по останній осі — експертах. Scores та softmax
#    обчислюйте у float32; probabilities поверніть у dtype вхідного `z`.
# 3. Для кожного токена знайдіть максимальну ймовірність і номер експерта.
#
# Поверніть `(probabilities, routes, gates)` форм `(B,L,E)`, `(B,L)`, `(B,L,1)`.
# Збережений вимір у gates потрібен, щоб множити на вихід FFN форми `(B,L,D)`.
# Не викликайте тут експертів: router лише вирішує, куди передати токен.
# Capacity mask, dispatch, експерти та повна мережа нижче вже надані.

# %%
def switch_router(z, router_weight):
    """Linear → softmax → top-1: поверніть probabilities, routes, gates."""
    ...


def test_switch_router():
    z = torch.tensor([[[2., 0.], [0., 2.]], [[0., 2.], [2., 0.]]])
    weight = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])
    probabilities, routes, gates = switch_router(z, weight)
    assert probabilities.shape == (2, 2, 3)
    assert routes.shape == (2, 2) and gates.shape == (2, 2, 1)
    assert routes.tolist() == [[0, 1], [1, 0]]
    expected = torch.tensor([0.8668133, 0.1173104, 0.0158762])
    torch.testing.assert_close(probabilities[0, 0], expected)
    torch.testing.assert_close(probabilities[0, 1], expected[[1, 0, 2]])
    torch.testing.assert_close(gates, torch.full((2, 2, 1), expected[0].item()))
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 2))
    # Softmax має бути стійким до великих scores та спільного зсуву всіх scores.
    shifted, _, _ = switch_router(z, weight + 1000.)
    torch.testing.assert_close(shifted, probabilities)
    # Усі експерти рівноймовірні: gate залишається 1/E, а не нормується до 1.
    _, _, uniform_gate = switch_router(z, torch.zeros_like(weight))
    torch.testing.assert_close(uniform_gate, torch.full_like(gates, 1 / 3))
    print("✓ router: probabilities, expert ids, gates and batch dimensions")

RUN_OPTIONAL_MOE = False
if __name__ == "__main__" and RUN_OPTIONAL_MOE:
    test_switch_router()

# %% [markdown]
# **Готова частина: dispatch і capacity — тут нічого дописувати не потрібно.**
# Checkpoint задає `expert_capacity=64`: у кожній послідовності експерт приймає
# перші 64 призначені токени. One-hot + cumsum рахує порядок призначень.
# Для точного відтворення Transformers 4.46.3 починаємо з копії z, замінюємо
# прийняті токени виходами експертів, потім множимо всі позиції на gate.
# Отже, overflow у цій реалізації дає `gate*z` усередині гілки.
# Зовнішній residual додається окремо. Це поведінка конкретної версії еталона.
# Router тут обробляє також padding; source padding маскується в attention.

# %%
def switch_moe(z, router_weight, experts, capacity):
    """Готовий dispatch: використайте студентський router і вже знайомі FFN."""
    _, routes, gates = switch_router(z, router_weight)
    one_hot = torch.nn.functional.one_hot(routes, num_classes=len(experts))
    priority = one_hot.cumsum(dim=1)
    dispatch = one_hot.bool() & (priority <= capacity)
    accepted = dispatch.any(dim=-1)
    next_states = z.clone()
    for expert_id, (wi, wo) in enumerate(experts):
        mask = dispatch[..., expert_id]
        if mask.any():
            next_states[mask] = ffn(z[mask], wi, wo)
    return gates * next_states, routes, accepted

# %% [markdown]
# ### 12.2. Збираємо повну мережу з уже реалізованих компонентів
#
# Цикл стеку надано: порівняйте його з вашим `transformer_stack`.
# В attention нічого не змінюється. Ваги FFN тепер мають префікс `mlp`.
# Якщо є `mlp.router.classifier.weight`, викликаємо `switch_moe`;
# інакше використовуємо звичайний dense `ffn`. Router і всі експерти
# беруть справжні навчені ваги з safetensors, без копіювання FFN T5-small.
# `routing_log` дозволяє побачити, які експерти працювали.

# %%
def switch_stack(ids, key_mask, weights, config, decoder=False, memory=None,
                 source_mask=None, routing_log=None):
    name = "decoder" if decoder else "encoder"
    x = embedding_lookup(ids, weights["shared.weight"])
    length = ids.shape[1]
    mask = attention_mask(key_mask, length, causal=decoder)
    bias = position_bias(weights[f"{name}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
                         length, length, not decoder, config.relative_attention_max_distance)
    n_layers = config.num_decoder_layers if decoder else config.num_layers
    for i in range(n_layers):
        base = f"{name}.block.{i}.layer."
        z = rms_norm(x, weights[base + "0.layer_norm.weight"], config.layer_norm_epsilon)
        x = residual_add(x, multi_head_attention(z, z, weights, base + "0.SelfAttention.",
                                               config.num_heads, mask, bias))
        if decoder:
            z = rms_norm(x, weights[base + "1.layer_norm.weight"], config.layer_norm_epsilon)
            cross_mask = attention_mask(source_mask, length)
            x = residual_add(x, multi_head_attention(z, memory, weights, base + "1.EncDecAttention.",
                                                   config.num_heads, cross_mask))
        ff = base + ("2." if decoder else "1.")
        z = rms_norm(x, weights[ff + "layer_norm.weight"], config.layer_norm_epsilon)
        mlp = ff + "mlp."
        if mlp + "router.classifier.weight" in weights:
            experts = [(weights[mlp + f"experts.expert_{j}.wi.weight"],
                        weights[mlp + f"experts.expert_{j}.wo.weight"])
                       for j in range(config.num_experts)]
            branch, routes, accepted = switch_moe(
                z, weights[mlp + "router.classifier.weight"], experts, config.expert_capacity)
            if routing_log is not None:
                routing_log.append((name, i, routes.clone(), accepted.clone()))
        else:
            branch = ffn(z, weights[mlp + "wi.weight"], weights[mlp + "wo.weight"])
        x = residual_add(x, branch)
    return rms_norm(x, weights[name + ".final_layer_norm.weight"], config.layer_norm_epsilon)


def switch_decode_logits(decoder_ids, memory, source_mask, weights, config, routing_log=None):
    hidden = switch_stack(decoder_ids, torch.ones_like(decoder_ids), weights, config,
                          decoder=True, memory=memory, source_mask=source_mask,
                          routing_log=routing_log)
    if config.tie_word_embeddings:
        hidden = hidden * config.d_model ** -0.5
    head = weights.get("lm_head.weight", weights["shared.weight"])
    return linear(hidden, head)


def switch_forward(input_ids, source_mask, decoder_ids, weights, config, routing_log=None):
    memory = switch_stack(input_ids, source_mask, weights, config, routing_log=routing_log)
    return switch_decode_logits(decoder_ids, memory, source_mask, weights, config, routing_log)


@torch.inference_mode()
def switch_generate(input_ids, source_mask, weights, config, max_new_tokens=24):
    if input_ids.shape[0] != 1:
        raise ValueError("Генеруємо по одному тексту")
    memory = switch_stack(input_ids, source_mask, weights, config)
    result = torch.full((1, 1), config.decoder_start_token_id, dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = switch_decode_logits(result, memory, source_mask, weights, config)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        result = torch.cat([result, next_id], dim=1)
        if next_id.item() == config.eos_token_id:
            break
    return result

# %% [markdown]
# ### 12.3. Швидка локальна перевірка перед завантаженням
#
# На випадковій маленькій моделі перевіряємо обчислення і переповнення capacity.
# Випадкові ваги тут — лише тест. Для осмисленого тексту далі завантажуємо
# повний pretrained Switch-base-8 checkpoint.

# %%
@torch.inference_mode()
def test_switch_model():
    torch.manual_seed(17)
    config = SwitchTransformersConfig(
        vocab_size=37, d_model=16, d_kv=4, d_ff=32, num_layers=2,
        num_decoder_layers=2, num_heads=4, num_experts=3,
        num_sparse_encoder_layers=1, num_sparse_decoder_layers=1,
        expert_capacity=1, dropout_rate=0., router_jitter_noise=0.,
        decoder_start_token_id=0, eos_token_id=1, pad_token_id=0)
    reference = SwitchTransformersForConditionalGeneration(config).eval()
    # Усі токени обирають expert_0: перевіряємо capacity overflow і невикликані експерти.
    for name, param in reference.named_parameters():
        if "router.classifier.weight" in name:
            param.zero_()
    weights = reference.state_dict()
    source = torch.tensor([[4, 5, 1, 0], [8, 9, 10, 1]])
    mask = source.ne(0).long()
    target = torch.tensor([[0, 6, 7], [0, 11, 12]])
    log = []
    actual = switch_forward(source, mask, target, weights, config, log)
    expected = reference(input_ids=source, attention_mask=mask,
                         decoder_input_ids=target, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert len(log) == 2
    for _, _, routes, accepted in log:
        assert (routes == 0).all()
        assert (accepted.sum(-1) == 1).all()
    # Інший router: більше одного активного експерта, без overflow.
    config.expert_capacity = 64
    reference = SwitchTransformersForConditionalGeneration(config).eval()
    weights = reference.state_dict()
    actual = switch_forward(source, mask, target, weights, config)
    expected = reference(input_ids=source, attention_mask=mask,
                         decoder_input_ids=target, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    ours = switch_generate(source[:1], mask[:1], weights, config, 4)
    theirs = reference.generate(input_ids=source[:1], attention_mask=mask[:1],
                                max_new_tokens=4, do_sample=False, num_beams=1, use_cache=False)
    assert torch.equal(ours, theirs)
    print("✓ Switch: dense/sparse blocks, router, capacity, logits and greedy ids")

if __name__ == "__main__" and RUN_OPTIONAL_MOE:
    test_switch_model()

# %% [markdown]
# ### 12.4. Завантажуємо навчені ваги та відновлюємо пропуски
#
# Увімкніть `RUN_OPTIONAL_MOE=True` після реалізації `switch_router` і запустіть
# комірки нижче. Download (~1.24 GB) та перетворення bf16 → float32 не входять
# у час inference. Еталон будуємо з тієї ж конфігурації та завантажуємо ті ж
# тензори. Власний forward не викликає готові Switch-блоки.

# %%
def load_switch_checkpoint():
    options = dict(repo_id=SWITCH_MODEL_ID, revision=SWITCH_REVISION, cache_dir=CACHE_DIR)
    config = SwitchTransformersConfig.from_json_file(hf_hub_download(filename="config.json", **options))
    assert config.router_bias is False and config.dense_act_fn == "relu"
    weights = load_file(hf_hub_download(filename="model.safetensors", **options), device="cpu")
    # Конвертуємо по одному тензору, звільняючи його bf16-версію.
    for name in weights:
        weights[name] = weights[name].float()
    # У safetensors tied-параметри можуть зберігатись лише під одним ім'ям.
    if "shared.weight" not in weights:
        weights["shared.weight"] = weights["encoder.embed_tokens.weight"]
    tokenizer = AutoTokenizer.from_pretrained(
        SWITCH_MODEL_ID, revision=SWITCH_REVISION, cache_dir=CACHE_DIR)
    return config, weights, tokenizer


def restore_single_span(prompt, generated_text):
    """Вставити перший згенерований span замість одного sentinel у prompt."""
    if prompt.count("<extra_id_0>") != 1 or "<extra_id_0>" not in generated_text:
        return None
    span = generated_text.split("<extra_id_0>", 1)[1]
    for marker in ("<extra_id_1>", "</s>", "<pad>"):
        span = span.split(marker, 1)[0]
    return prompt.replace("<extra_id_0>", span.strip())


@torch.inference_mode()
def compare_switch_pretrained(config, weights, tokenizer):
    # Завантаження тих самих ваг без другої мережевої копії.
    reference = SwitchTransformersForConditionalGeneration(config).eval()
    complete = dict(weights)
    if config.tie_word_embeddings:
        for name in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"):
            complete[name] = weights["shared.weight"]
    reference.load_state_dict(complete, strict=True)
    results = []
    for prompt in ("The capital of France is <extra_id_0>.",
                   "The opposite of hot is <extra_id_0>."):
        inputs = tokenizer(prompt, return_tensors="pt")
        prefix = torch.tensor([[config.decoder_start_token_id, tokenizer.convert_tokens_to_ids("<extra_id_0>")]])
        log = []
        actual = switch_forward(inputs.input_ids, inputs.attention_mask, prefix, weights, config, log)
        expected = reference(**inputs, decoder_input_ids=prefix, use_cache=False).logits
        error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
        ours = switch_generate(inputs.input_ids, inputs.attention_mask, weights, config)
        theirs = reference.generate(**inputs, max_new_tokens=24, do_sample=False,
                                    num_beams=1, use_cache=False)
        assert torch.equal(ours, theirs), (ours, theirs)
        raw = tokenizer.decode(ours[0], skip_special_tokens=False)
        restored = restore_single_span(prompt, raw)
        print("Вхід:", prompt)
        print("Decoder:", raw)
        print("Відновлений текст:", restored if restored is not None else "Не отримано sentinel-відповідь")
        print(f"✓ max |Δlogits|={error:.2e}; greedy id збігаються")
        for stack_name, layer, routes, accepted in log:
            if stack_name == "encoder":
                valid = inputs.attention_mask.bool() & accepted
                counts = torch.bincount(routes[valid], minlength=config.num_experts)
                print(f"  encoder block {layer}: {counts.tolist()} токенів на експерта")
        results.append(dict(prompt=prompt, generated=raw, restored=restored, max_logit_error=error))
    return results

if __name__ == "__main__" and RUN_OPTIONAL_MOE:
    switch_config, switch_weights, switch_tokenizer = load_switch_checkpoint()
    switch_results = compare_switch_pretrained(switch_config, switch_weights, switch_tokenizer)

# %% [markdown]
# **Приклад перевіреного запуску** з наведеним snapshot у float32:
#
# | Вхід | Відновлений текст |
# |---|---|
# | `The capital of France is <extra_id_0>.` | `The capital of France is Paris.` |
# | `The opposite of hot is <extra_id_0>.` | `The opposite of hot is cold.` |
#
# Для обох прикладів власний forward дав ті самі logits і greedy id,
# що й Transformers 4.46.3. Це короткі демонстрації span filling;
# вони не гарантують правильного відновлення довільних фактів чи речень.
#
# ### 12.5. Необов'язковий звіт
#
# 1. Відновіть пропуски у трьох власних англійських реченнях з одним sentinel.
#    Запишіть input, decoder output зі спеціальними токенами та відновлений текст.
# 2. Поясніть, чому це інша задача, ніж переклад у частині T5-small.
# 3. Наведіть похибку logits і перевірку збігу greedy id з Transformers.
# 4. Для справжніх токенів encoder побудуйте гістограму призначень експертам.
#    Поясніть, чому різні sparse-блоки можуть мати різні маршрути.
# 5. Що станеться, якщо нормувати top-1 gate до 1? Перевірте зміну logits.
# 6. Чому sparse-модель має багато параметрів, хоча токен проходить через
#    один експерт на sparse-блок? Чому Python dispatch не гарантує прискорення?
#
# %% [markdown]
# ## 13. Дослідження та звіт
#
# 1. Намалюйте шлях source → encoder → cross-attention → decoder → vocabulary.
#    Позначте форми для source довжини 7 і target довжини 4.
# 2. Поясніть, які саме функції зі своїх попередніх робіт ви адаптували.
# 3. Тимчасово додайте ділення scores на sqrt(d_kv). Як змінилися похибка
#    logits і текст? Відновіть правильну реалізацію.
# 4. Вимкніть causal mask. Чи проходить тест зміни майбутнього токена?
# 5. Порівняйте переклади трьох власних речень; збережіть prompt, id і текст.
# 6. Поясніть, чому токенізатор із власним словником несумісний із цими вагами.
# 7. Де саме обчислення повторюються без KV-cache? Що можна було б кешувати?
#
# Здайте notebook із заповненими функціями, виконаними перевірками,
# результатами перекладу, таблицею часу для трьох режимів та короткими відповідями. Очікуваний результат —
# усі локальні перевірки проходять, pretrained logits близькі до еталона,
# greedy id збігаються для наведених прикладів.
#
# ## Джерела
#
# - [T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
# - [Checkpoint і конфігурація T5-small](https://huggingface.co/google-t5/t5-small)
# - [Документація T5](https://huggingface.co/docs/transformers/model_doc/t5)
# - [Еталонна реалізація Transformers 4.46.3](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/t5/modeling_t5.py)
