# Полный пайплайн `compute_metrics_v3.ipynb`

## 1) Входные данные и файлы

Ноутбук стартует с чтения файлов:

- `out/meta.json`
- `out/emb_FRIDA.npy`

Пути заданы явно:

- `BASE = Path('out')`
- `META_PATH = BASE / 'meta.json'`
- `EMB_PATH = BASE / 'emb_FRIDA.npy'`
- `REPORT_PATH = BASE / 'document_metrics_report_v3.json'`

Проверка на старте:

- `len(meta)` должен совпадать с `emb.shape[0]`, иначе ошибка `Mismatch`.

---

## 2) Константы и пороги (все значения)

### Ограничения вывода

- `TOP_N_TRANSITIONS = 7`
- `TOP_N_OUTLIERS = 7`
- `TOP_N_ISSUES = 12`

### Пороги метрик

- `REDUNDANCY_THRESHOLD = 0.90`
- `COVERAGE_THRESHOLD = 0.72`
- `NOVELTY_LOW_THRESHOLD = 0.58`
- `NOVELTY_HIGH_THRESHOLD = 0.88`
- `ADJ_WEAK_Z = -1.0`
- `OUTLIER_Z = -1.0`

### Пороги глобальных флагов (интерпретация)

- `adj_mean < 0.65` -> `low_local_coherence`
- `coverage_score < 0.50` -> `low_goal_coverage`
- `conclusion_delta < -0.05` -> `conclusion_main_misalignment`
- `redund_90 > 0.25` -> `high_redundancy`
- `jump_share > 0.30` -> `high_semantic_jumps`

### Градация severity для локальных проблем

- для `weak_transition`:
  - `adj_z <= -1.5` -> `high`
  - `-1.5 < adj_z <= -1.0` -> `medium`
  - иначе `low`
- для `outlier_segment`:
  - `centrality_z <= -1.5` -> `high`
  - `-1.5 < centrality_z <= -1.0` -> `medium`
- для `weak_section_boundary`:
  - `z <= -1.5` -> `high`
  - `-1.5 < z <= -1.0` -> `medium`

---

## 3) Подготовка структуры документа

Из `meta.json` берутся:

- `roles` (например `intro`, `main`, `conclusion`)
- `paths` (путь/идентификатор сегмента)
- `parents` (родительский раздел)

Формируются индексы:

- `main_idx`, `intro_idx`, `conc_idx`
- `intro_i` и `conc_i` (первый индекс intro/conclusion)

Также собирается группировка `parent -> [indices]`.

---

## 4) Базовая матрица сходства

Строится косинусная матрица:

1. Нормализация эмбеддингов по L2.
2. `S = vectors @ vectors.T`

`S[i, j]` используется почти во всех последующих метриках как источник сигналов.

---

## 5) Метрики: что и как считается

## 5.1 Локальная связность (`local_coherence`)

- `adj[i] = S[i, i+1]` для соседних сегментов.
- `adj_mean`, `adj_std`.
- `adj_z = (adj - adj_mean) / adj_std`.
- `weak_transition_indices`: где `adj_z <= ADJ_WEAK_Z`.
- `weakest_transitions_top_n`: худшие переходы (до `TOP_N_TRANSITIONS`).

Выходные поля:

- `adj_mean`
- `adj_std`
- `weak_transition_count`
- `weak_transition_share`
- `weakest_transitions_top_n`

## 5.2 Структурная согласованность (`structure_consistency`)

- `main_main_mean`: средняя попарная схожесть внутри `main` (верхний треугольник без диагонали).
- `intro_to_main_mean`: средняя схожесть intro -> main.
- `conclusion_to_main_mean`: средняя схожесть conclusion -> main.
- `intro_delta = intro_to_main_mean - main_main_mean`
- `conclusion_delta = conclusion_to_main_mean - main_main_mean`

## 5.3 Избыточность `main` (`redundancy`)

- `redund_90 = доля(main-main пар, где similarity > 0.90)`

## 5.4 Дрейф и гладкость (`drift_and_smoothness`)

- Шаговые векторы: `step_vecs = emb[i+1] - emb[i]`
- `step_mean_norm = mean(||step_vecs||)`
- `turn_mean_sharpness = mean(1 - cos(step_i, step_{i+1}))`
- `semantic_drift_score = 0.6 * step_mean_norm + 0.4 * turn_mean_sharpness`

Гладкость:

- `second_diff_mean = mean(||emb[i+2] - 2*emb[i+1] + emb[i]||)`
- `skip2_mean = mean(S[i, i+2])`
- `smoothness_score = 0.5 * (1 / (1 + second_diff_mean)) + 0.5 * skip2_mean`

## 5.5 Центральность и выбросы (`centrality`)

- Диагональ удаляется из расчета:
  - `S_wo_diag = S`, затем `diag = NaN`
- `centrality_all[i] = nanmean(S_wo_diag[i, :])`
- `centrality_z`
- `outlierness = -centrality_z`
- `top_outliers`: топ сегментов с наибольшей `outlierness` (до `TOP_N_OUTLIERS`)

## 5.6 Coverage (`coverage`)

Для каждого intro-сегмента:

- лучший матч с `main`: `max(S[intro, main])`
- лучший матч с `conclusion`: `max(S[intro, conclusion])`

Далее:

- `coverage_main = доля intro, где best_intro_main >= COVERAGE_THRESHOLD`
- `coverage_conclusion = доля intro, где best_intro_conclusion >= COVERAGE_THRESHOLD`
- `coverage_score = среднее по доступным частям`

## 5.7 Профиль новизны/повтора (`novelty_redundancy_profile`)

Для каждого сегмента `i > 0`:

- `sim_prev = max(S[i, :i])`
- если `sim_prev >= 0.88` -> `redundant`
- если `sim_prev <= 0.58` -> `jump`
- иначе `balanced`

Итоги:

- `redundant_share`
- `jump_share`
- `items` (по каждому сегменту)

## 5.8 Переходы между разделами (`section_boundaries`)

- Если `parents[i] != parents[i+1]`, фиксируется boundary-переход.
- Для него сохраняются similarity и `z`.

---

## 6) Интерпретация: флаги и локальные проблемы

Формируются:

- `global_flags` (по порогам из раздела 2)
- `localized_issues`:
  - `weak_transition`
  - `outlier_segment`
  - `weak_section_boundary`

`localized_issues` сортируется по серьезности (`high` -> `medium` -> `low`) и обрезается до `TOP_N_ISSUES`.

---

## 7) Итоговая агрегированная оценка `risk_score_v3`

Формула:

- `+ max(0, 0.70 - adj_mean) * 100`
- `+ min(20, semantic_drift_score * 15)`
- `+ max(0, 0.60 - smoothness_score) * 35`
- `+ max(0, 0.70 - coverage_score) * 45`
- `+ max(0, redund_90 - 0.10) * 60`
- `+ jump_share * 30`
- затем ограничение в диапазон `[0, 100]`

Важно:

- `risk_score_v3` в ноутбуке помечен как вспомогательный.
- Главная интерпретация идет через `localized_issues` и `global_flags`.

---

## 8) Финальный JSON-отчет (структура)

Сохраняется в `out/document_metrics_report_v3.json`:

- `version`, `model`, `n_segments`
- `diagnostic_metrics`
  - `local_coherence`
  - `structure_consistency`
  - `drift_and_smoothness`
  - `centrality`
  - `coverage`
  - `novelty_redundancy_profile`
  - `section_boundaries`
  - `redundancy`
- `interpretation`
  - `global_flags`
  - `localized_issues`
- `summary`
  - `risk_score_v3`
  - `note`

---

## 9) Текущие фактические результаты (по вашим двум прогонам v3)

Источник:

- `out/document_metrics_report_v3_orig.json`
- `out/document_metrics_report_v3_test.json`

### 9.1 Оригинальный документ (`v3_orig`)

- `risk_score_v3 = 16.6465`
- `adj_mean = 0.7163`
- `semantic_drift_score = 0.9922`
- `smoothness_score = 0.5496`
- `coverage_score = 1.0`
- `conclusion_delta = 0.0839`
- `redund_90 = 0.0`
- `jump_share = 0.0`
- `global_flags_count = 0`
- `localized_issues_count = 10`

### 9.2 Измененный документ (`v3_test`)

- `risk_score_v3 = 45.6724`
- `adj_mean = 0.5731`
- `semantic_drift_score = 1.1328`
- `smoothness_score = 0.4859`
- `coverage_score = 0.5`
- `conclusion_delta = -0.0504`
- `redund_90 = 0.0`
- `jump_share = 0.1`
- `global_flags_count = 2`
- `localized_issues_count = 11`

Сработавшие глобальные флаги в `v3_test`:

- `low_local_coherence`
- `conclusion_main_misalignment`

Итог по сравнению `orig` vs `test`:

- при намеренной подмене частей курсовой система повышает риск и ловит несогласованность структуры/связности;
- это ожидаемое и корректное поведение для текущей версии пайплайна.

