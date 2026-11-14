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
10. Необов’язково: MoE замість FFN — Linear router, sparse top-k dispatch,
    зважена сума експертів і підключення до останнього encoder-блока.
    Експерти повторно використовують реалізований FFN; навчання не потрібне.
    Щоб виконати додаткові комірки, заповніть `moe_ffn` та встановіть
    `RUN_OPTIONAL_MOE = True`. Основна частина не залежить від цієї вправи.

Для підтримки матеріалів:

```bash
make figures          # відтворити SVG-схему MoE
make notebook         # відтворити notebook із percent-cell Python
make check            # локальні перевірки на еталонних підстановках
make check-pretrained # додатково завантажити T5 і перевірити реальні переклади
```

Перевірки не заповнюють студентський файл. `tools/reference_bodies.json`
містить еталонні тіла вправ для викладацької перевірки. У власній реалізації
не використовуємо готові T5-блоки: Transformers служить токенізатором та еталоном.
