# Стилометр: паспорт и контрольный эксперимент

## Назначение

Модуль измеряет близость русской художественной прозы к паспорту автора. Здесь нет rewriter, LLM-judge, retrieval или генеративного цикла: сначала проверяется измеримость стиля на данных, не участвовавших в обучении.

## Установка и команды

До завершения TASK-002 зависимости этой задачи объявлены отдельно, поскольку `pyproject.toml` находится вне `_Boundary`:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r stylometry/requirements-dev.txt
.venv/bin/python -m stylometry --help
```

Паспорт одного автора:

```bash
.venv/bin/python -m stylometry card \
  --corpus data/stylometry/corpus/chekhov \
  --author chekhov \
  --golden data/stylometry/golden/masked.jsonl \
  --output data/stylometry/manual/style.json
```

Скорер текста:

```bash
.venv/bin/python -m stylometry score \
  --passport data/stylometry/baseline/style.json \
  --input path/to/text.txt
```

Контрольный эксперимент:

```bash
.venv/bin/python -m stylometry experiment \
  --corpus data/stylometry/corpus/chekhov \
  --other data/stylometry/corpus/dostoevsky \
  --other data/stylometry/corpus/technical \
  --golden data/stylometry/golden/masked.jsonl \
  --baseline-report data/stylometry/baseline/experiment.json \
  --report-version 1.0.0 \
  --output data/stylometry/reproduction
```

При повторении baseline без смены порогов версия остаётся `1.0.0`. Опубликованный `data/stylometry/baseline/experiment.json` используется валидатором автоматически, даже если `--baseline-report` опущен. Если изменён `auc_min` или `mannwhitney_p_max`, требуется большая `report_version`; `tuning_split=held_out` запрещён программно. Просмотр held_out-результата не является основанием подгонять пороги.

## Дисциплина данных

- Файл — минимальная единица сплита. Последние `ceil(25%)` отсортированных файлов каждого корпуса образуют held_out, остальные — train. Чанки никогда не пересекают границу файла и не перекрываются.
- Паспорт строится только на train-файлах автора после quarantine golden-файлов. Для baseline используются неперекрывающиеся окна по 200 слов.
- Дискриминатор: TF-IDF char n-gram 3–5 и бинарный `LogisticRegression`, `random_state=42`. Обучение — train обоих классов, AUC — только held_out-чанки.
- Паспортное расстояние: среднее `|z|` по всем `FEATURE_NAMES`; mean/std вычисляются на train-паспорте. Нулевая дисперсия получает безопасный масштаб `1.0`.
- Разделение расстояний проверяется `scipy.stats.mannwhitneyu(..., alternative="less")`: расстояния held_out-чанков автора должны быть меньше чужих.
- Дрейф содержит среднее расстояние для каждого файла автора, включая held_out и quarantine-файлы. Эти файлы не попадают в паспорт или обучение из-за расчёта дрейфа.

## Golden и человеческий review-gate

`data/stylometry/golden/masked.jsonl` содержит явную `schema_version` для каждой записи. Любая запись немедленно исключает весь `source_file` из паспорта и обучения. В отчётные golden-метрики попадает только запись со всеми полями:

```json
"review": {
  "reviewer": "идентификатор человека",
  "status": "approved",
  "reviewed_at": "RFC-3339 timestamp"
}
```

Исходные baseline-записи намеренно не имеют `review`: агент не имитирует человеческое подтверждение. Они учитываются в quarantine и перечисляются в `excluded_unreviewed_ids`, но не имеют метрик. Маскированные тексты хранятся отдельными копиями; исходный корпус не изменяется.

## Артефакты

Все JSON/JSONL имеют версию схемы `1.0`:

- `style.json` — все признаки, mean/std, число чанков, train- и quarantine-файлы;
- `drift.json` — средние расстояния всех файлов автора;
- `experiment.json` — сплиты, AUC, Mann-Whitney, review-gate, пороги и итоги приёмки;
- `sources.json` — URL и происхождение каждого корпусного файла;
- `baseline_config.json` — зафиксированные до контрольного прогона параметры.

Исходные тексты — паблик-домен произведения Чехова и Достоевского и статьи дореволюционного Энциклопедического словаря Брокгауза и Ефрона (технический антифон). Точные страницы указаны в `data/stylometry/sources.json`. Повторное получение выполняется явной командой `python scripts/stylometry/fetch_wikisource.py`; тесты и эксперимент сеть не используют.

## Baseline 1.0.0

Контрольный прогон выполнен на 804 train-чанках и 795 held_out-чанках. Паспорт построен на 229 чанках из трёх train-файлов Чехова; `02_step.txt` исключён по golden quarantine. Held_out автора — целые файлы `05_duel.txt` и `06_tri_goda.txt`.

| Метрика | Результат | Замороженный порог | Итог |
|---|---:|---:|---|
| AUC char 3–5 LogReg | 0.9897646557764437 | ≥ 0.90 | passed |
| Mann-Whitney, alternative=less | p = 3.2686575708710404e-40 | < 0.01 | passed |

Все три JSON-артефакта побайтово воспроизведены повторным прогоном. До человеческого review golden-набор не даёт отдельной оценки маскирования; это не влияет на AUC и Mann-Whitney, рассчитанные на независимом held_out. Подробности — в `data/stylometry/baseline/experiment.json`.
