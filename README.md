# Synthetic Healthcare Dataset: Lung Cancer Risk

A Faker-based pipeline that generates a large synthetic patient dataset (~10 million rows), then deliberately corrupts a copy of it to produce a **messy** dataset and stores that in a compressed Parquet file.

- `synthetic_healthcare_dataset.py`: the generator (clean data, rule-based diagnosis)
- `generated_data_merge.ipynb`: the full pipeline (generation → clean CSV → noise injection → messy CSV → Parquet)

> All data is entirely synthetic. It is intended for learning, prototyping, and testing data pipelines or ML models. It must not be used for clinical decisions.

---

## 1. Industry Domain and Use Case

**Domain:** Healthcare (clinical risk screening for lung cancer).

**Use case:** Practice and testing data for a data science workflow, in two stages:

1. **Clean dataset:** patient visit records with a `Positive`/`Negative` lung cancer diagnosis label, suitable for binary classification and exploratory analysis.
2. **Messy dataset:** a corrupted copy with missing values, inconsistent spellings, invalid entries, and mixed types, suitable for practicing data cleaning, validation, and ETL at a scale (10M+ rows) where chunked processing matters.

Real patient data cannot be shared freely because of privacy rules, so a synthetic dataset with realistic risk patterns (smoking, age, pollution, genetics, symptoms) is used instead.

---

## 2. Schema Design

Fourteen columns per record. "Parquet type" is the type enforced when the messy dataset is written to Parquet.

| # | Column | Generated type | Parquet type | Allowed / expected values | Description |
|---|---|---|---|---|---|
| 1 | `Patient_id` | string | string | `PTN-<number>-<year>` | Patient identifier; number increases per record, suffix is the visit year |
| 2 | `Record_id` | string | string | `RCD-<number>-<year>` | Record identifier, same numbering scheme as `Patient_id` |
| 3 | `branch_id` | string | string | `NE-1565`, `NW-6754`, `SE-6239`, `SW-4543` | Hospital branch |
| 4 | `Visit_date` | string | string | `D-MM-YYYY`, 2016–2025 | Visit date (day is not zero-padded) |
| 5 | `Age` | int | string | 1–100 | Patient age. Stored as string in Parquet because noisy values can be text |
| 6 | `Gender` | string | string | `Male`, `Female` | Patient gender |
| 7 | `Smoking_status` | string | string | `Former`, `Current`, `Never` | Smoking history |
| 8 | `Pack_years` | int | string | 0, 15–25, 25–30 | Smoking exposure; depends on `Smoking_status`. Stored as string for the same reason as `Age` |
| 9 | `Air_pollution_exposure` | string | string | `Very low`, `Low`, `Medium`, `High`, `Very high` | Environmental exposure level |
| 10 | `Genetic_risk` | string | string | `Low`, `Medium`, `High` | Genetic predisposition |
| 11 | `Chronic_lung_disease` | int (0/1) | float64 | 0, 1 | Pre-existing chronic lung disease flag |
| 12 | `Chest_pain` | int (0/1) | float64 | 0, 1 | Symptom flag |
| 13 | `Shortness_of_breath` | int (0/1) | float64 | 0, 1 | Symptom flag |
| 14 | `Diagnosis_result` | string | string | `Positive`, `Negative` | Label derived from a risk score |

The three flag columns use `float64` in Parquet so that injected missing values (NaN) can be represented.

---

## 3. Probability Distributions, Constraints, and Noise

### 3.1 Probability distributions

Every categorical field is sampled with fixed weights through Faker's `random_element`, so weights act as relative probabilities.

| Field | Distribution |
|---|---|
| `branch_id` | Uniform: 25% each branch |
| Visit year | 2016: 3%, 2017: 4%, 2018: 6%, 2019: 9%, 2020: 9%, 2021: 11%, 2022: 12%, 2023: 13%, 2024: 15%, 2025: 18% (recent years more likely) |
| Visit month | Jan: 12%, Feb: 7%, Mar: 6%, Apr: 6%, May: 7%, Jun: 7%, Jul: 6%, Aug: 6%, Sep: 7%, Oct: 8%, Nov: 14%, Dec: 14% (winter peak) |
| Visit day | Uniform over the valid days of the month |
| `Age` | Bracket first, then uniform within it: 1–20: 0.07%, 20–34: 0.19%, 35–44: 1.94%, 45–54: 8.83%, 55–64: 20.97%, 65–74: 31.86%, 75–84: 28.92%, 85–100: 7.22% (skewed old, matching typical lung cancer screening populations) |
| `Gender` | Male 55.34%, Female 44.66% |
| `Smoking_status` | Former 50.83%, Current 36.72%, Never 12.47% |
| `Pack_years` | Uniform within a range set by smoking status (see constraints) |
| `Air_pollution_exposure` | Very low 6.15%, Low 18.40%, Medium 34.72%, High 28.61%, Very high 12.12% |
| `Genetic_risk` | Low 63.40%, Medium 28.15%, High 8.45% |
| `Chest_pain` | 1 with weight 0.2743, 0 with 0.7257 (~27.4% positive) |
| `Shortness_of_breath` | 1 with weight 0.3819, 0 with 0.6181 (~38.2% positive) |
| `Chronic_lung_disease` | Weights are both `0.1029`, which normalizes to **50/50** (see Known Caveats) |

### 3.2 Realistic constraints

- **Pack-years depend on smoking status:** `Former` → 15–25, `Current` → 25–30, `Never` → exactly 0.
- **Valid calendar days:** 31 days for Jan/Mar/May/Jul/Aug/Oct/Dec, 30 for Apr/Jun/Sep/Nov, and 28 for February (leap years ignored).
- **Consistent years:** the year in `Patient_id` / `Record_id` matches the year in `Visit_date`.
- **Unique, increasing IDs:** the numeric part increases with each record and across batches. In parallel mode each chunk starts at `1000001 * (seed + 1)` so chunks use different ranges.
- **Diagnosis is not random:** `Diagnosis_result` is computed from the other columns by a risk score.
  - **Baseline points:** age bracket, male gender, smoking status and pack-years, pollution exposure, genetic risk, chronic lung disease, and each symptom.
  - **Interaction rules (A–G):** examples are both symptoms together (+1.8), a non-smoker with high pollution and high genetic risk (+2.5), a current smoker with high pollution and chronic lung disease (+2.5), and protective buffers for young, asymptomatic patients.
  - **Threshold:** `Positive` when the score is **≥ 6.5**, otherwise `Negative`.
- In a 200,000-row test run, about **78% of records were `Positive`**, so the label is imbalanced toward the positive class. Adjust the threshold (`self.threshold`) if a more balanced dataset is needed.

### 3.3 Noise injected (messy dataset)

Noise is added in `generated_data_merge.ipynb` using a seeded NumPy generator (`np.random.default_rng(67)`), so it is reproducible. The clean CSV is reloaded first, then corrupted.

**Missing values:** each of these columns independently has **2%** of its values set to null: `Age`, `Gender`, `Smoking_status`, `Pack_years`, `Air_pollution_exposure`, `Genetic_risk`, `Chest_pain`, `Chronic_lung_disease`, `Shortness_of_breath`.

**Inconsistent categorical spellings:**

| Field | Applied to | Probability | Replacement values |
|---|---|---|---|
| `Gender` | rows that are `Female` | 2% | `F`, `female`, `fem`, `Female ` (trailing space) |
| `Smoking_status` | rows that are `Current` | 3% | `Before`, `ForM3R`, `former`, `FORmER ` |
| `Smoking_status` | rows that are `Never` | 7% | `didnt`, `NEv`, `Negative`, `None `, `No`, `NEvER` |
| `Smoking_status` | rows that are `Former` | 10% | `CUR`, `Current `, `CURrENT`, `NOW ` |
| `Smoking_status` | rows that are `Male` (see caveat) | 2% | `M`, `male`, `masc`, `Male ` |

**Invalid or mixed-type values:**

| Field | Probability | Replacement values |
|---|---|---|
| `Pack_years` | 0.1% of all rows | `-28`, `350`, `"0"`, `"Twenty three"` (negative, impossible, numeric-as-text, and spelled-out numbers) |
| `Age` | 0% (mask is defined but disabled) | `-15`, `"SIx seven"`, `"69"`, `"Twenty Five"` |
| `Visit_date` | 0.1% of all rows | `10-10-2000` (out of range), `31-12-2026` (future date), `October 8, 2024` (different format) |

Typical cleaning tasks this enables: trimming whitespace, normalizing case, mapping variants to canonical categories, imputing nulls, range validation for `Pack_years` and `Age`, and parsing mixed date formats.

---

## 4. Storage Footprint Analysis

**How these numbers were obtained:** the pipeline was run end to end on a 200,000-row sample (Python 3.12, pandas 3.0.2, pyarrow, single vCPU sandbox), file sizes were measured, and results were scaled linearly to the full **10,000,233 rows**. Row size is stable because every column is short, so the extrapolation is reliable to within a few percent. Re-run the full pipeline to confirm exact figures on your machine.

### File sizes

| File | Format | Measured (200,000 rows) | Bytes per row | Estimated (10,000,233 rows) |
|---|---|---|---|---|
| `updated_synthetic_lung_cancer_dataset_a_1.csv` | CSV, clean (uncompressed) | 19.6 MB | ~98 | **~980 MB** |
| `messy_lung_cancer_dataset.csv` | CSV, messy (uncompressed) | 21.4 MB | ~107 | **~1.07 GB** |
| `parquet_messy_lung_cancer_dataset.parquet` | Parquet, Snappy compressed | 3.7 MB | ~18.5 | **~185 MB** |

- **Total uncompressed size:** ~1.0 GB for the clean CSV and ~1.07 GB for the messy CSV.
- **Compression:** Snappy-compressed Parquet is about **5.8× smaller** than the messy CSV (~83% reduction). Parquet stores columns separately and dictionary-encodes repeated strings (smoking status, pollution level, branch), which is why the categorical-heavy data compresses so well.
- The messy CSV is ~9% larger than the clean one mainly because null injection turns integer columns into floats (e.g. `69` becomes `69.0`) and the corrupted values are longer.
- The Parquet file is written in 100,000-row chunks, so peak memory stays low during export.

### Generator runtime

| Measurement | Result |
|---|---|
| Iterative generation, 200,000 rows (1 vCPU) | 7.7 s (~25,900 rows/s) |
| Iterative generation, 10,000,233 rows (estimated) | ~6.5 minutes on a similar single core |
| Parallel generation (`joblib`) | Scales with core count: expect roughly (single-core time ÷ cores). Run the script to see the printed `parallel: Time of execution` value |
| Your machine (fill in after a full run) | parallel: ____ s, iterative: ____ s |

The script and notebook print both timings along with the dataset shape and the number of `Patient_id` collisions.

---

## 5. Requirements and Usage

- Python 3.12+ (the script uses nested same-type quotes inside f-strings)

```bash
pip install faker pandas numpy joblib pyarrow
```

**Run the generator only:**

```bash
python synthetic_healthcare_dataset.py
```

**Run the full pipeline:** run the cells of `generated_data_merge.ipynb` in order (generate → export clean CSV → inject noise → export messy CSV → write Parquet).

### Configuration

| Setting | Where | Default | Description |
|---|---|---|---|
| `df_length` | module level | `10000233` | Total number of rows to generate |
| `num_chunks` | module level | `os.cpu_count() * 2` | Number of parallel chunks |
| `dataset_length` | `Healthcare.__init__` | `300` | Rows per provider instance |
| `batch_size` | `Healthcare.__init__` | `0.10` | Fraction of rows per batch |
| `start_id` | `Healthcare.__init__` | `1000001` | Starting number for ID generation |
| `threshold` | `Healthcare.__init__` | `6.5` | Risk score needed for a `Positive` diagnosis |
| `default_rng(67)` | notebook | `67` | Seed for noise injection |

---

## 6. Known Caveats

- **Chronic lung disease probability:** `cld_p_true` and `cld_p_false` are both `0.1029`, which gives roughly 50% prevalence instead of ~10%. If ~10% is intended, set `cld_p_false` to `0.8971`.
- **Null genetics:** `Genetic_risk` is never `"Null"` in the generated data, so the null-handling branch and Rule G never trigger.
- **Gender noise writes to the wrong column:** in the notebook, `male_mask` assigns `M`/`male`/`masc`/`Male ` to `Smoking_status` instead of `Gender`.
- **Smoking status noise looks swapped:** `Current` rows receive "former"-style typos and `Former` rows receive "current"-style typos.
- **pandas 3.x:** assigning strings into the numeric `Pack_years` / `Age` columns raises a `TypeError`. Cast them first with `df[["Pack_years", "Age"]] = df[["Pack_years", "Age"]].astype(object)`.
- **ID collisions in parallel mode:** chunks start IDs 1,000,001 apart, so a chunk with more than ~1M rows can overlap with the next. Check the printed collision count.
- **Reproducibility:** only the parallel mode is seeded; the iterative mode is not.
- **Dates:** February is capped at 28 days.
- **Memory:** ~10M rows in a DataFrame (twice, if both modes are run) needs substantial RAM. Lower `df_length` for testing.

---

## Project Structure

```
.
├── synthetic_healthcare_dataset.py   # Generator script
├── generated_data_merge.ipynb        # Full pipeline: generate, add noise, export
└── README.md
```
