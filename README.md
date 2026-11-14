# Фінальна лабораторна: Transformer encoder–decoder (T5)

Продовження [lab-tensors](https://github.com/lpnu-llm/lab-tensors),
[lab-tokenizer](https://github.com/lpnu-llm/lab-tokenizer),
[lab-attention](https://github.com/lpnu-llm/lab-attention),
[lab-positional-encoding](https://github.com/lpnu-llm/lab-positional-encoding)
та [Lab-encoder](https://github.com/lpnu-llm/Lab-encoder).
Адаптуємо раніше реалізовані функції до тензорів, збираємо T5-small,
завантажуємо ваги Hugging Face та генеруємо переклад власним forward.

Відкрийте [lab_transformer.ipynb](lab_transformer.ipynb) у Jupyter або VS Code.
Заповнюйте `...` і запускайте комірки по черзі. Python-джерело:
[lab_transformer.py](lab_transformer.py). Потрібен Python 3.10+ та залежності:

```bash
python3 -m pip install -r requirements.txt
```

У notebook можна виконати `%pip install -r requirements.txt`.
CPU достатньо; T5-small має близько 60 млн параметрів (приблизно 240 MB
float32 ваг). Перший запуск завантажує ваги й токенізатор у `.cache/huggingface`;
наступні використовують кеш. Перевірка з еталоном тримає дві копії ваг у RAM.

Порядок роботи:

1. Перенесення embedding, Linear, residual, RMSNorm та FFN із попередніх робіт.
2. Padding/causal маски й relative position bias T5.
3. Multi-head self-attention та cross-attention.
4. Encoder із двома та decoder із трьома pre-norm гілками.
5. Shared embeddings, фінальна нормалізація та LM head.
6. Перевірка logits на маленькій випадковій моделі без мережі.
7. Завантаження safetensors, greedy decoding, зупинка на EOS.
8. Порівняння pretrained logits і token id із Transformers; власні переклади.
9. Швидкість генерації: власна реалізація, Transformers без cache та з KV-cache;
   прогрів, медіана часу й tokens/s для однакових виходів.
10. Необов’язково: справжній pretrained MoE `google/switch-base-8`.
    Реалізуємо Linear router (softmax і top-1) та множення виходу експерта
    на його router-ймовірність; dispatch надано. Використовуємо
    T5-компоненти, завантажуємо safetensors і відновлюємо пропуски в тексті.
    Навчання не потрібне. Заповніть `switch_router`, `apply_router_gate` та встановіть
    `RUN_OPTIONAL_MOE = True`. Основна частина не залежить від цієї вправи.
    Завантаження: приблизно 1.24 GB bf16 ваг; CPU float32 копія — 2.48 GB,
    для порівняння з еталоном потрібні дві копії та додаткова пам’ять.

Для підтримки матеріалів:

```bash
make figures          # перевірити локальний оригінал рисунка зі статті
make notebook         # відтворити notebook із percent-cell Python
make check            # локальні перевірки на еталонних підстановках
make check-pretrained # додатково завантажити T5 і перевірити реальні переклади
make check-switch-pretrained # завантажити Switch (1.24 GB) і перевірити span filling
```

Перевірки не заповнюють студентський файл. `tools/reference_bodies.json`
містить еталонні тіла вправ для викладацької перевірки. У власній реалізації
не використовуємо готові T5-блоки: Transformers служить токенізатором та еталоном.

T5-small не містить MoE. Додаткове завдання збирає іншу навчену модель —
Switch-base-8 зі справжніми router-вагами та експертами. Вона відновлює
пропущені фрагменти англійського тексту; це не chat-модель і не готовий перекладач.
Оригінальний рисунок 2 зі статті Switch Transformers показує її top-1 routing.
[Походження рисунка](figures/README.md).

Перевірений приклад MoE: `The capital of France is <extra_id_0>.` →
`The capital of France is Paris.`; `The opposite of hot is <extra_id_0>.` →
`The opposite of hot is cold.` Власні logits та greedy id збіглися з Transformers.
