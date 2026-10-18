# Synthetic Healthcare Dataset: Lung Cancer Risk

A Faker-based pipeline that generates a large synthetic patient dataset (~5 million rows), corrupts it into a **messy** version, then runs a cleaner over the messy data. Messy and cleaned outputs are streamed to Parquet.

> All data is entirely synthetic. Use it for learning, prototyping, and testing data pipelines or ML models. Never for clinical decisions.

## Files

| File | Role |
|---|---|
| `synthetic_healthcare_dataset.py` | `Healthcare` Faker provider: generates clean records and the rule-based diagnosis |
| `clean_to_messy.py` | `DirtyDataInjector`: adds nulls, typos, and invalid values |
| `data_cleaning.py` | `LungCancerCleaner`: fixes the messy data step by step |
| `data_pipeline_generator.py` | Main entry point: parallel generate → corrupt → clean → Parquet |

```
.
├── synthetic_healthcare_dataset.py
├── clean_to_messy.py
├── data_cleaning.py
├── data_pipeline_generator.py
└── README.md
```

## How the pipeline works

1. Rows are split into `os.cpu_count() * 2` chunks. Each chunk runs in its own `joblib` worker with its own seed.
2. Each worker generates batches (10% of the chunk each). For every batch:
   - `DirtyDataInjector` corrupts a copy of the raw batch → appended to `messy_folder/file-<seed>.csv`.
   - `LungCancerCleaner` cleans that messy batch → appended to `cleaned_folder/file-<seed>.csv`.
3. Polars lazily scans each folder and streams it into one Parquet file, so the full dataset never sits in memory.

| Output | Content |
|---|---|
| `messy_data.parquet` | Corrupted data, every column read as text |
| `cleaned_data.parquet` | Output of the cleaner (types inferred from the first 10,000 rows) |

The "clean" output is the **cleaned messy data**, not the raw generator output. Cleaning is lossy (see Caveats).

## Schema

14 columns per generated record.

| Column | Type | Values | Notes |
|---|---|---|---|
| `Patient_id` | string | `PTN-<number>-<year>` | Number increases per record; year = visit year |
| `Record_id` | string | `RCD-<number>-<year>` | Same scheme as `Patient_id` |
| `branch_id` | string | `NE-1565`, `NW-6754`, `SE-6239`, `SW-4543` | Hospital branch |
| `Visit_date` | string | `D-MM-YYYY`, 2016–2025 | Day not zero-padded |
| `Age` | int | 1–100 | |
| `Gender` | string | `Male`, `Female` | |
| `Smoking_status` | string | `Former`, `Current`, `Never` | |
| `Pack_years` | int | 0–60 | Depends on smoking status and age |
| `Air_pollution_exposure` | string | `Very low` … `Very high` | 5 levels |
| `Genetic_risk` | string | `Low`, `Medium`, `High` | |
| `Chronic_lung_disease` | int | 0, 1 | |
| `Chest_pain` | int | 0, 1 | |
| `Shortness_of_breath` | int | 0, 1 | |
| `Diagnosis_result` | string | `Positive`, `Negative` | Derived from a risk score |

After cleaning, the three flag columns become `Yes`/`No`, `Visit_date` becomes a datetime, and `Age`/`Pack_years` are numeric.

## Data generation

### Distributions

Categorical fields use fixed weights via Faker's `random_element`.

| Field | Distribution |
|---|---|
| `branch_id` | 25% each |
| Visit year | 2016: 3%, 2017: 4%, 2018: 6%, 2019: 9%, 2020: 9%, 2021: 11%, 2022: 12%, 2023: 13%, 2024: 15%, 2025: 18% |
| Visit month | Jan 12%, Feb 7%, Mar 6%, Apr 6%, May 7%, Jun 7%, Jul 6%, Aug 6%, Sep 7%, Oct 8%, Nov 14%, Dec 14% |
| Visit day | Uniform over valid days (February capped at 28) |
| `Age` | Bracket, then uniform within it: 1–20: 0.07%, 21–34: 0.19%, 35–44: 1.94%, 45–54: 8.83%, 55–64: 20.97%, 65–74: 31.86%, 75–84: 28.92%, 85–100: 7.22% |
| `Gender` | Male 55.34%, Female 44.66% |
| `Smoking_status` | Former 50.83%, Current 36.72%, Never 12.47% |
| `Air_pollution_exposure` | Very low 6.15%, Low 18.40%, Medium 34.72%, High 28.61%, Very high 12.12% |
| `Genetic_risk` | Low 63.40%, Medium 28.15%, High 8.45% |
| `Chronic_lung_disease` | 1: 10.29% |
| `Chest_pain` | 1: 27.43% |
| `Shortness_of_breath` | 1: 38.19% |

### Constraints

- **Pack-years depend on smoking status and age:**

  | Group | Pack-years |
  |---|---|
  | Under 16 (always `Never`) or `Never` | 0 |
  | `Former` | 1–40 |
  | `Current` | 10–60 |
  | Ages 17–23, `Former` | 1–10 |
  | Ages 17–23, `Current` | 0–20 |

- **Consistent years:** the year in `Patient_id` / `Record_id` matches `Visit_date`.
- **Unique IDs:** numbers increase across records and batches. Each chunk starts at `1000001 * (seed + 1)`.
- **Diagnosis is computed, not random:** a risk score is built from baseline points (age, gender, smoking, pack-years, pollution, genetics, chronic lung disease, symptoms) plus interaction rules A–G (for example, both symptoms together +1.8, and protective buffers for young asymptomatic patients). `Positive` when the score is **≥ 9.5** (`self.threshold`).
- The positive rate has not been re-measured at this threshold (an earlier run at 6.5 gave ~78% positive). Tune `threshold` to change class balance.

## Noise injection (messy data)

Implemented in `clean_to_messy.py`. Each batch gets its own seed (`seed * 1000 + batch_index`), so runs are reproducible. Summary below is carried over from the earlier notebook version; check it against `clean_to_messy.py`.

| Type | Details |
|---|---|
| Missing values | 2% nulls in each of `Age`, `Gender`, `Smoking_status`, `Pack_years`, `Air_pollution_exposure`, `Genetic_risk`, and the three flag columns |
| `Gender` typos | `Female` rows (2%) → `F`, `female`, `fem`, `Female ` |
| `Smoking_status` typos | `Current` (3%), `Never` (7%), `Former` (10%) rows get misspelled or wrong-category variants (`ForM3R`, `didnt`, `None `, `CUR`, `NOW `, …) |
| `Smoking_status` gender values | `Male` values (`M`, `male`, `masc`) written into `Smoking_status` |
| `Pack_years` | 0.1% invalid: `-28`, `350`, `"0"`, `"Twenty three"` |
| `Visit_date` | 0.1% invalid: out of range, future date, or a different format (`October 8, 2024`) |
| `Age` | Corruption defined but disabled |

## Cleaning steps (`LungCancerCleaner`)

Runs in this order on a copy of the input:

1. **Numeric:** convert `Age` and `Pack_years` to numbers (text becomes null); set `Age` outside 0–120 to null; fill nulls with the column median.
2. **Categorical:** strip, title-case, and fill nulls with `Unknown` (`Gender`, `Smoking_status`, `Air_pollution_exposure`, `Genetic_risk`).
3. **Binary:** map 0/1 to `No`/`Yes`; nulls temporarily `Unknown`.
4. **Dates:** parse `Visit_date` (day first, mixed formats); drop rows with a year of 2027 or later (2026 is kept).
5. **Labels:** map `Gender` and `Smoking_status` variants to standard values; set gender-like values in `Smoking_status` to `Unknown`; map `Negative`/`No`/`Didnt`/`None` to `Never`.
6. **Binary imputation:** replace `Unknown` flags with the most common known answer.

## Requirements and usage

Python 3.12+ (nested same-type quotes in f-strings).

```bash
pip install faker pandas numpy joblib polars pyarrow
python data_pipeline_generator.py
```

The generator alone can be run with `python synthetic_healthcare_dataset.py`. It benchmarks parallel vs. iterative generation of 1,000,233 rows and prints shape, duplicate `Patient_id` count, and runtime.

### Configuration

| Setting | Where | Default | Description |
|---|---|---|---|
| `DF_LENGTH` | `data_pipeline_generator.py` | `5004574` | Total rows |
| `MIN_CHUNK_ROWS` | `data_pipeline_generator.py` | `10` | Minimum rows per chunk |
| `CLEAN_DIR` / `MESSY_DIR` | `data_pipeline_generator.py` | `cleaned_folder` / `messy_folder` | Per-chunk CSV folders (emptied on each run) |
| `CLEAN_PARQUET` / `MESSY_PARQUET` | `data_pipeline_generator.py` | `cleaned_data.parquet` / `messy_data.parquet` | Final outputs |
| `dataset_length` | `Healthcare.__init__` | `300` | Rows per provider instance |
| `batch_size` | `Healthcare.__init__` | `0.10` | Fraction of rows per batch |
| `start_id` | `Healthcare.__init__` | `1000001` | First ID number |
| `threshold` | `Healthcare.__init__` | `9.5` | Score needed for `Positive` |

## Storage estimate

Based on an earlier 200,000-row run of the messy output, scaled linearly. Re-run the pipeline for exact figures.

| File | Bytes per row | Estimated (5,004,574 rows) |
|---|---|---|
| Messy CSV (per-chunk files) | ~107 | ~535 MB |
| Messy Parquet (Snappy) | ~18.5 | ~93 MB |

Parquet is roughly 5.8× smaller than the messy CSV because repeated strings are dictionary-encoded.

## Known caveats

- **Cleaning is per batch:** `LungCancerCleaner` runs on each batch, so medians and "most common" binary answers are computed per batch, not over the full dataset.
- **Date filter is loose:** only visits from 2027 onward are dropped, so the injected future date `31-12-2026` survives. Unparseable dates are kept as null.
- **Invalid values survive cleaning:** out-of-range `Pack_years` (`-28`, `350`) and old dates (`10-10-2000`) are not removed. Only `Age` is range-checked.
- **ID columns are kept:** the drop step in `LungCancerCleaner.drop_columns` is commented out.
- **Null genetics:** `Genetic_risk` is never `"Null"` in generated data, so that branch and Rule G never trigger.
- **Gender noise:** `Male` variants are written to `Smoking_status` instead of `Gender` (the cleaner compensates by setting them to `Unknown`).
- **Smoking noise looks swapped:** `Current` rows receive "former"-style typos and `Former` rows receive "current"-style typos.
- **Smoking map typo:** `SMOKING_MAP` contains `nvever` (likely meant `never`). It does not matter because plain `never` is also mapped.
- **Stale docstrings:** `generate_status_pack` still describes the old pack-year ranges (15–25 / 25–30), and the `clean_numeric` docstring still says only negative ages are removed.
- **pandas 3.x:** assigning strings into numeric columns raises `TypeError`; cast to `object` first.
- **Reproducibility:** `Faker` and NumPy are seeded per chunk. Iterative mode in the generator's `__main__` is not seeded.
- **Memory (generator `__main__`):** concatenating all chunks in memory needs substantial RAM; the main pipeline avoids this by streaming.
