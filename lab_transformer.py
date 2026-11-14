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
# ## 12. Необов'язково: збираємо Mixture-of-Experts
#
# Усі компоненти експерта вже знайомі: це та сама `ffn(z, wi, wo)`.
# Новий навчуваний компонент — **router / gating network**: один Linear
# без bias із `D` входами та `E` виходами, по одному score на експерта.
# Softmax перетворює scores на ймовірності. Router працює **окремо для кожного
# токена**, використовуючи його поточний прихований стан, а не id слова.
#
# Це не останній шар моделі перед словником. MoE замінює FFN-гілку обраного
# Transformer-блока. Attention, RMSNorm, residual і LM head уже реалізовані.
# Для першого експерименту замініть лише FFN останнього encoder-блока:
#
# ![MoE: router обирає два FFN-експерти та змішує їхні виходи](figures/moe.svg)
#
# Приклад на рисунку: softmax дає `[0.50, 0.20, 0.30]`; top-2 обирає
# експерти 1 і 3. Після нормування: `0.50/0.80=0.625`, `0.30/0.80=0.375`.
#
# Для `z=(B,L,D)`: `scores=linear(z, router_weight)` має форму `(B,L,E)`.
# Оберіть `k` найбільших імовірностей; перенормуйте **тільки обрані** ваги
# так, щоб їхня сума дорівнювала 1. Потім обчисліть
# `y = sum(gate_j * expert_j(z))` для обраних експертів.
# Тут `E=3`, за замовчуванням `k=2`; можливий також `k=1`.
# Це навчальний варіант top-k routing, не точна реалізація Switch Transformer:
# у Switch top-1 вихід масштабується початковою ймовірністю router.
# У нашому нормованому top-1 вага завжди 1; для навчання router через task loss
# цього недостатньо. Навчання, capacity limits і балансувальні loss тут не додаємо.
# [Джерело: Switch Transformers](https://jmlr.org/papers/v23/21-0998.html).
#
# **Вправа.** Реалізуйте `moe_ffn`, повторно використовуючи `linear` та `ffn`.
# `experts` — список пар `(wi, wo)`, кожна пара має власні ваги.
# Поверніть `(y, indices, gates)` форм `(B,L,D)`, `(B,L,k)`, `(B,L,k)`.
# Зведіть токени до `(B*L,D)`, для кожного експерта виберіть лише призначені
# йому токени, викличте FFN для цієї підмножини та додайте зважені виходи
# на початкові позиції. Не обчислюйте всі експерти для всіх токенів:
# це перевірить формулу, але не реалізує sparse execution.
# При `k < 1` або `k > E` підніміть `ValueError`. Невибраний експерт не викликайте.

# %%
def moe_ffn(z, router_weight, experts, top_k=2):
    """Sparse top-k FFN; ваги gates нормовані серед обраних експертів."""
    ...


def moe_residual(x, gamma, eps, router_weight, experts, top_k=2):
    z = rms_norm(x, gamma, eps)
    branch, indices, gates = moe_ffn(z, router_weight, experts, top_k)
    return residual_add(x, branch), indices, gates


@torch.inference_mode()
def test_moe():
    z = torch.tensor([[[2., 1.], [1., 3.]], [[4., 1.], [1., 2.]]])
    eye = torch.eye(2)
    experts = [(eye, eye), (eye, 2 * eye), (eye, 3 * eye)]
    router = torch.tensor([[2., 0.], [0., 2.], [-1., -1.]])
    for k in (1, 2, 3):
        actual, indices, gates = moe_ffn(z, router, experts, k)
        torch.testing.assert_close(gates.sum(-1), torch.ones_like(z[..., 0]))
        assert indices.shape == gates.shape == (*z.shape[:-1], k)
        # Незалежна dense-формула лише для перевірки sparse-реалізації.
        probabilities = torch.softmax(z @ router.T, -1)
        chosen, expected_indices = probabilities.topk(k, -1)
        chosen = chosen / chosen.sum(-1, keepdim=True)
        assert torch.equal(indices, expected_indices)
        torch.testing.assert_close(gates, chosen)
        expected = torch.zeros_like(z)
        for expert_id, (wi, wo) in enumerate(experts):
            mass = ((indices == expert_id) * gates).sum(-1, keepdim=True)
            expected += mass * ffn(z, wi, wo)
        torch.testing.assert_close(actual, expected)
    # Третій експерт ніколи не обирається при k=2; його NaN не мають потрапити у вихід.
    unused = experts[:2] + [(eye, torch.full_like(eye, float("nan")))]
    assert torch.isfinite(moe_ffn(z, router, unused, 2)[0]).all()
    identical = [experts[0]] * 3
    torch.testing.assert_close(moe_ffn(z, router, identical, 2)[0], ffn(z, eye, eye))
    output, _, _ = moe_residual(z, torch.ones(2), 1e-6, router, identical)
    torch.testing.assert_close(output, z + ffn(rms_norm(z, torch.ones(2), 1e-6), eye, eye))
    for k in (0, 4):
        try:
            moe_ffn(z, router, experts, k)
        except ValueError:
            pass
        else:
            raise AssertionError("Перевірте межі top_k")
    print("✓ optional MoE: routing, weighted sum, unused expert, residual")

# Цей розділ можна пропустити, не заповнюючи moe_ffn.
RUN_OPTIONAL_MOE = False
if __name__ == "__main__" and RUN_OPTIONAL_MOE:
    test_moe()

# %% [markdown]
# ### Підключаємо до зібраного T5
#
# Зробіть окрему копію `transformer_stack` під назвою `moe_transformer_stack`.
# Залиште основну T5-реалізацію для порівнянь. Після attention останнього
# encoder-блока замість його RMSNorm + FFN + residual викличте `moe_residual`.
# Передайте gamma з `encoder.block.5.layer.1.layer_norm.weight` та три
# **окремі клони** його `DenseReluDense.wi.weight` і `wo.weight`.
# Router ініціалізуйте `torch.randn(3, config.d_model) * 0.01` із фіксованим seed.
# Інші блоки та фінальну RMSNorm виконайте без змін. Потрібні нові аргументи
# `router_weight` і `experts` для копії стеку. Підключення всередині циклу:
#
# ```python
# if not decoder and i == config.num_layers - 1:
#     x, routes, gates = moe_residual(
#         x, weights[ff + "layer_norm.weight"], config.layer_norm_epsilon,
#         router_weight, experts, top_k=2)
# else:
#     # Попередня RMSNorm → FFN → residual гілка.
#     ...
# ```
#
# 1. З однаковими експертами порівняйте стани encoder із початковим T5:
#    вони мають збігатися в межах float32 похибки для будь-якого routing,
#    бо сума gates дорівнює 1. Це перевірка правильності підключення.
# 2. Лише для експерименту змініть `wo` другого експерта, наприклад помножте
#    на 1.1. Перевірте, що стани змінюються для токенів, які його обирають.
#    Такі вручну змінені ваги не є навченими MoE-експертами; поліпшення
#    перекладу не очікуємо. T5-small checkpoint не містить MoE router чи експертів.
# 3. Порахуйте `torch.bincount(routes.reshape(-1), minlength=3)` для справжніх
#    токенів джерела, виключивши padding. Чи всі експерти отримують токени?
# 4. Порівняйте кількість параметрів: dense FFN має `2*D*d_ff`, MoE має
#    `E*2*D*d_ff + E*D`; на токен виконується `k` FFN і router.
#    Більша загальна кількість параметрів не означає менший час. Наші Python
#    dispatch-цикли можуть бути повільними; прискорення тут не обіцяємо.
#
# Для необов'язкового звіту достатньо реалізації, локальних перевірок,
# порівняння encoder-станів і гістограми призначень. Навчати MoE не потрібно.

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
