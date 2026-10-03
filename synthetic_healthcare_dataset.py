"""Synthetic lung-cancer-themed healthcare dataset generator.

This module defines a custom Faker provider, ``Healthcare``, that produces
synthetic patient records (demographics, lifestyle, environment, genetics,
symptoms) and assigns a ``Positive``/``Negative`` diagnosis using a rule-based
risk score. It then benchmarks two ways of generating ~10 million rows:

    1. Parallel: rows are split into chunks and generated across CPU cores
       with joblib, each chunk using its own random seed.
    2. Iterative: all rows are generated in a single process.

All data is entirely synthetic and must not be used for clinical decisions.

Typical usage example:

    python synthetic_healthcare_dataset.py
"""

print('Running...\n\n\n\n')

# --- Imports ---------------------------------------------------------------
from faker import Faker                       # Fake data generator
from faker.providers import BaseProvider      # Base class for custom providers
import pandas as pd                           # DataFrame construction
import numpy as np                            # Seeding numpy's RNG
from collections import OrderedDict           # Ordered {value: weight} maps
from joblib import delayed, Parallel          # Multi-process generation
import time                                   # Benchmark timing
import os                                     # CPU count lookup


class Healthcare(BaseProvider):
  """Faker provider that generates synthetic lung-cancer-themed patient data.

  The provider is iterable. Each iteration yields a dictionary of equal-length
  lists (one list per column), which represents one *batch* of records and can
  be passed directly to ``pd.DataFrame``.

  Attributes:
    dataset_length: Total number of records to generate.
    batch_length: Number of records generated per batch.
    threshold: Risk score at or above which a record is labelled ``Positive``.
    cld_p_true: Weight for chronic lung disease being present (1).
    cld_p_false: Weight for chronic lung disease being absent (0).
    cp_p_true: Weight for chest pain being present (1).
    cp_p_false: Weight for chest pain being absent (0).
    sb_p_true: Weight for shortness of breath being present (1).
    sb_p_false: Weight for shortness of breath being absent (0).
    cache_year_list: Years sampled for the current batch. Reused so that IDs
      and visit dates share the same year.
  """

  def __init__(self, generator, dataset_length=300, batch_size=0.10, start_id=1000001):
    """Initializes the provider.

    Args:
      generator: The ``Faker`` instance this provider is attached to. It is
        used as the random source for all weighted sampling.
      dataset_length: Total number of records to generate.
      batch_size: Fraction of the dataset per batch (0.10 means 10%). It is
        converted internally to a percentage (``batch_size * 100``).
      start_id: First numeric value used when building patient and record IDs.
    """
    super().__init__(generator)
    self._INC_DIGIT_STRING = start_id          # Running base for ID numbers
    self._batch_tracker = 0                    # Records already emitted
    self._batch_size = batch_size * 100        # Batch size as a percentage

    self.dataset_length = dataset_length
    # Number of records per batch (integer division by the percentage value).
    self.batch_length = int(self.dataset_length // self._batch_size)
    # Minimum risk score required for a "Positive" diagnosis.
    self.threshold = 6.5

    # Weights for binary (0/1) features. Faker normalizes these weights, so
    # they do not need to sum to 1.
    self.cld_p_true, self.cld_p_false = 0.1029, 0.1029   # Chronic lung disease
    self.cp_p_true, self.cp_p_false = 0.2743, 0.7257     # Chest pain
    self.sb_p_true, self.sb_p_false = 0.3819, 0.6181     # Shortness of breath

  def __iter__(self):
    """Iterates over the dataset one batch at a time.

    Each batch is built column by column. The year list is generated first
    because the IDs and visit dates depend on it.

    Yields:
      A dict mapping column name to a list of values, with one entry per
      record in the batch. Columns are ``Patient_id``, ``Record_id``,
      ``branch_id``, ``Visit_date``, ``Age``, ``Gender``, ``Smoking_status``,
      ``Pack_years``, ``Air_pollution_exposure``, ``Genetic_risk``,
      ``Chronic_lung_disease``, ``Chest_pain``, ``Shortness_of_breath`` and
      ``Diagnosis_result``.
    """
    remaining = self.dataset_length
    while remaining > 0:
      # The last batch may be smaller than the standard batch length.
      self.batch_length = min(remaining, self.batch_length)

      # Years must be generated first: IDs and dates reuse them.
      self.cache_year_list = self._generate_year_distribution()
      # Smoking status and pack-years are generated together because
      # pack-years depends on the status.
      smoking_status_data, pack_years_data = self.generate_status_pack()
      customer = {
        'Patient_id':self.patient_record_id(id_type='patient'),
        'Record_id':self.patient_record_id(id_type='record'),
        'branch_id':self.generate_branch_id(),
        'Visit_date':self.generate_date(),
        'Age':self.generate_age(),
        'Gender':self.generate_gender(),
        'Smoking_status':smoking_status_data,
        'Pack_years':pack_years_data,
        'Air_pollution_exposure':self.generate_exposure(),
        'Genetic_risk':self.generate_genetic_risks(),
        'Chronic_lung_disease':self.generate_binary_data(p_true=self.cld_p_true,p_false=self.cld_p_false),
        'Chest_pain':self.generate_binary_data(p_true=self.cp_p_true,p_false=self.cp_p_false),
        'Shortness_of_breath':self.generate_binary_data(p_true=self.sb_p_true,p_false=self.sb_p_false)
    }

      # The label is computed last, from all the other columns in the batch.
      customer['Diagnosis_result'] = self.assign_diagnosis(customer)

      # Advance the counters so the next batch gets fresh, non-overlapping IDs.
      self._batch_tracker += self.batch_length
      remaining -= self.batch_length

      yield customer

  def _generate_year_distribution(self) -> None:
    '''Generates a weighted list of visit years for the current batch.

    Also defines the month weights used later by ``generate_date``. Weights
    skew toward recent years and toward November to January.

    Returns:
      A list of year strings (e.g. ``'2024'``) of length ``batch_length``.
      Note: the return annotation says ``None`` but a list is returned.
    '''
    # Probability of each year (2016-2025); later years are more likely.
    self._year_weights = OrderedDict([
        ('2016',0.03),('2017',0.04),('2018',0.06),('2019',0.09),('2020',0.09),('2021',0.11),('2022',0.12),('2023',0.13),('2024',0.15),('2025',0.18)
    ])
    # Probability of each month; winter months (Nov-Jan) are more likely.
    self._month_weights = OrderedDict([
        ('01',0.12),('02',0.07),('03',0.06),('04',0.06),('05',0.07),('06',0.07),('07',0.06),('08',0.06),('09',0.07),('10',0.08),('11',0.14),('12',0.14)
    ])
    return [self.generator.random_element(self._year_weights) for _ in range(self.batch_length)]

  def patient_record_id(self, id_type='patient') -> list[str]:
    """Builds unique patient or record IDs for the current batch.

    The ID format is ``<PREFIX>-<number>-<year>``, where the number increases
    with each record and across batches.

    Args:
      id_type: ``'patient'`` for ``PTN-`` IDs or ``'record'`` for ``RCD-`` IDs.

    Returns:
      A list of ID strings, one per record in the batch.

    Raises:
      ValueError: If ``id_type`` is not ``'patient'`` or ``'record'``.
    """
    patient_list = list()

    if id_type not in ['patient', 'record']:
      raise ValueError('Input for id_type is invalid')

    for i,yr in enumerate(self.cache_year_list): # iteration are depended on the ouput of private function
      prefix = 'PTN' if id_type=='patient' else 'RCD'
      # Number = base + position in batch + records from earlier batches.
      patient_list.append(f"{prefix}-{self._INC_DIGIT_STRING  + i + self._batch_tracker}-{yr}")

    return patient_list

  def generate_branch_id(self):
    """Randomly assigns each record to a hospital branch.

    All four branches have equal probability.

    Returns:
      A list of branch ID strings (e.g. ``'NE-1565'``).
    """
    self._branches = OrderedDict([
        ('NE-1565',0.25),('NW-6754',0.25),('SE-6239',0.25),('SW-4543',0.25)
    ])
    return [self.generator.random_element(self._branches) for _ in range(self.batch_length)]

  def generate_date(self):
    """Generates visit dates consistent with the cached years.

    The month is sampled using the month weights, then a random valid day is
    chosen. February is capped at 28 days for simplicity (no leap years).

    Returns:
      A list of date strings formatted as ``D-MM-YYYY``.
    """
    date_list = list()

    for i in range(self.batch_length):
      month = self.generator.random_element(self._month_weights)
      year = self.cache_year_list[i]   # Same year used in the record's IDs

      # Pick the day range based on the number of days in the month.
      if month in ['01','03','05','07','08','10','12']:
        day = self.generator.random_int(min=1,max=31)
      elif month in ['04','06','09','11']:
        day = self.generator.random_int(min=1,max=30)
      else:
        day = self.generator.random_int(min=1,max=28)

      date_list.append(f'{day}-{month}-{year}')

    return date_list

  def generate_age(self):
    """Generates patient ages from a weighted age-bracket distribution.

    A bracket is sampled first (weighted toward 55 to 84), then a uniform
    random age is drawn within the bracket's bounds.

    Returns:
      A list of integer ages, one per record.
    """
    age_list = list()
    # Bracket labels are "min-max" strings; they are sliced below to get bounds.
    self._age_weights = OrderedDict([
        ('01-20',0.0007),('20-34',0.0019),('35-44',0.0194),('45-54', 0.0883),('55-64',0.2097),('65-74',0.3186),('75-84',0.2892),('85-100',0.0722)
    ])

    for _ in range(self.batch_length):
      random_age_category = self.generator.random_element(self._age_weights)
      # [:2] is the lower bound and [3:] is the upper bound of the label.
      age_list.append(
          self.generator.random_int(min=int(random_age_category[:2]), max=int(random_age_category[3:]))
      )

    return age_list

  def generate_gender(self):
    """Randomly assigns a gender to each record.

    Returns:
      A list of ``'Male'`` (55.34%) or ``'Female'`` (44.66%) strings.
    """
    self._gender_weights = OrderedDict([
        ('Male', 0.5534),('Female',0.4466)])
    return [self.generator.random_element(self._gender_weights) for _ in range(self.batch_length)]

  def generate_status_pack(self):
    """Generates smoking status together with matching pack-years.

    Pack-years depend on the status, so both columns are produced in one pass:
    ``Former`` gets 15 to 25, ``Current`` gets 25 to 30, ``Never`` gets 0.

    Returns:
      A tuple ``(status_list, pack_list)`` where ``status_list`` contains
      ``'Former'``, ``'Current'`` or ``'Never'`` and ``pack_list`` contains the
      corresponding integer pack-years.
    """
    status_list = list()
    pack_list = list()
    self._smoking_weights = OrderedDict([
        ('Former',0.5083),('Current',0.3672),('Never',0.1247)
    ])

    for _ in range(self.batch_length):
      random_status = str(self.generator.random_element(self._smoking_weights))

      # Pack-years are tied to the sampled smoking status.
      if random_status == 'Former':
        packs = self.generator.random_int(min=15,max=25)
      elif random_status == 'Current':
        packs = self.generator.random_int(min=25,max=30)
      else:
        packs = 0

      status_list.append(random_status)
      pack_list.append(packs)


    return status_list, pack_list

  def generate_exposure(self):
    """Randomly assigns an air pollution exposure level to each record.

    Returns:
      A list of strings: ``'Very low'``, ``'Low'``, ``'Medium'``, ``'High'``
      or ``'Very high'``.
    """
    self._pollution_exposure = OrderedDict([
        ('Very low', 0.0615),('Low', 0.1840), ('Medium',0.3472),('High',0.2861),('Very high',0.1212)
    ])
    return [self.generator.random_element(self._pollution_exposure) for _ in range(self.batch_length)]

  def generate_genetic_risks(self):
    """Randomly assigns a genetic risk level to each record.

    Returns:
      A list of strings: ``'Low'`` (63.4%), ``'Medium'`` (28.15%) or
      ``'High'`` (8.45%).
    """
    self._genetics = OrderedDict([
        ('Low',0.6340),('Medium',0.2815),('High',0.0845)
    ])
    return [self.generator.random_element(self._genetics) for _ in range(self.batch_length)]

  def generate_binary_data(self, p_true, p_false, value_true=1, value_false=0):
    """Generates a weighted binary column (for example, a symptom flag).

    Args:
      p_true: Weight of the "true" value.
      p_false: Weight of the "false" value.
      value_true: Value stored when the feature is present.
      value_false: Value stored when the feature is absent.

    Returns:
      A list of ``value_true``/``value_false`` entries, one per record.
    """
    self.hashmap = OrderedDict([
        (value_true,p_true),(value_false,p_false)
    ])
    return [self.generator.random_element(self.hashmap) for _ in range(self.batch_length)]

  def assign_diagnosis(self, row):
    """Assigns a diagnosis to each record using a rule-based risk score.

    For every record, a numeric score is built in three stages:

      1. Baseline points from age, gender, smoking, pollution, genetics,
         chronic lung disease, and symptoms.
      2. Interaction rules (A to G) that add or subtract points for specific
         combinations of factors.
      3. A threshold check: the record is ``Positive`` if the score is at or
         above ``self.threshold``, otherwise ``Negative``.

    Args:
      row: A dict of column lists for the current batch, as built in
        ``__iter__``. It must contain ``Age``, ``Gender``, ``Smoking_status``,
        ``Pack_years``, ``Air_pollution_exposure``, ``Genetic_risk``,
        ``Chronic_lung_disease``, ``Chest_pain`` and ``Shortness_of_breath``.

    Returns:
      A list of ``'Positive'`` or ``'Negative'`` strings, one per record.
    """
    score_list = list()
    for i in range(self.batch_length):
      score = 0.0

      # ---------------------------------------------------------
      # 1. BASELINE RISK SCORES
      # ---------------------------------------------------------

      # Age Brackets: risk rises with age; the youngest groups lower the score.
      if row["Age"][i] > 0 and row['Age'][i] < 21:
          score -= 3.0
      elif row["Age"][i] > 20 and row['Age'][i] < 31:
          score -= 2.0
      elif row["Age"][i] > 30 and row['Age'][i] < 41:
          score += 0.5
      elif row["Age"][i] > 40 and row['Age'][i] < 51:
          score += 1.5
      elif row["Age"][i] > 50 and row['Age'][i] < 61:
          score += 2.0
      elif row["Age"][i] > 60:
          score += 2.5

      # Gender (Slight statistical baseline offset)
      if row["Gender"][i] == "Male":
          score += 0.3

      # Smoking Status & Pack Years: heavier exposure adds a bonus on top of
      # the base points for the status.
      if row["Smoking_status"][i] == "Current":
          score += 3.0
          if row["Pack_years"][i] >= 25:
              score += 1.5
      elif row["Smoking_status"][i] == "Former":
          score += 1.5
          if row["Pack_years"][i] >= 15:
              score += 1.0
      elif row["Smoking_status"][i] == "Never":
          score -= 1.0

      # Air Pollution Exposure
      if row["Air_pollution_exposure"][i] == "Very high":
          score += 2.5
      elif row["Air_pollution_exposure"][i] == "High":
          score += 1.5
      elif row["Air_pollution_exposure"][i] == "Medium":
          score += 0.5
      elif row["Air_pollution_exposure"][i] == "Very low":
          score -= 0.5

      # Genetics Risk (Including Null handling)
      if row["Genetic_risk"][i] == "High":
          score += 3.0
      elif row["Genetic_risk"][i] == "Medium":
          score += 1.2
      elif row["Genetic_risk"][i] == "Low":
          score -= 0.5
      elif row["Genetic_risk"][i] == "Null" or pd.isna(row["Genetic_risk"]):
          # Impute/penalize based on unknown background
          score += 0.5

      # Chronic Lung Disease
      if row["Chronic_lung_disease"][i] == 1:
          score += 2.0

      # Individual Symptoms
      if row["Chest_pain"][i] == 1:
          score += 1.2
      if row["Shortness_of_breath"][i] == 1:
          score += 1.2

      # ---------------------------------------------------------
      # 2. ADVANCED INTERACTION & SPECIFIC CASE RULES
      # ---------------------------------------------------------

      # Rule A: Symptom Synergy (Co-occurrence of both symptoms is significantly worse)
      if row["Chest_pain"][i] == 1 and row["Shortness_of_breath"][i] == 1:
          score += 1.8

      # Rule B: Non-Smoker High Environmental + Genetic Risk (Targeted driver for non-smokers)
      if (
          row["Smoking_status"][i] == "Never"
          and row["Air_pollution_exposure"][i] in ["High", "Very high"]
          and row["Genetic_risk"][i] == "High"
      ):
          score += 2.5

      # Rule C: The "Triple Compound" High Risk (Smoking + Pollution + Pre-existing Disease)
      if (
          row["Smoking_status"][i] == "Current"
          and row["Air_pollution_exposure"][i] in ["High", "Very high"]
          and row["Chronic_lung_disease"][i] == 1
      ):
          score += 2.5

      # Rule D: Former Smoker Vulnerability in Older Age
      if (
          row["Smoking_status"][i] == "Former"
          and row["Age"][i] > 65
          and row["Chronic_lung_disease"][i] == 1
      ):
          score += 1.5

      # Rule E: "Clean Bill" Protection (Young + Never Smoker + Low Exposure + No Symptoms)
      if (
          row["Age"][i] > 0 and row['Age'][i] < 35
          and row["Smoking_status"][i] == "Never"
          and row["Air_pollution_exposure"][i] in ["Low", "Very low"]
          and row["Chest_pain"][i] == 0
          and row["Shortness_of_breath"][i] == 0
      ):
          score -= 3.5

      # Rule F: Young Asymptomatic Smoker Buffer (Prevents immediate positive label for young heavy smokers without symptoms)
      if (
          row["Age"][i] > 0 and row['Age'][i] < 35
          and row["Chest_pain"][i] == 0
          and row["Shortness_of_breath"][i] == 0
          and row["Genetic_risk"][i] in ["Low", "Medium"]
      ):
          score -= 1.5

      # Rule G: Null Genetics Fallback Hazard (Adds risk if Genetics is missing AND severe factors exist)
      if (
          (row["Genetic_risk"][i] == "Null")
          and row["Smoking_status"][i] == "Current"
          and row["Chronic_lung_disease"][i] == 1
      ):
          score += 1.0

      # ---------------------------------------------------------
      # 3. FINAL DIAGNOSIS THRESHOLD
      # ---------------------------------------------------------
      score_list.append("Positive" if score >= self.threshold else "Negative")

    return score_list


start = time.perf_counter()

def generate_data(n, seed):
  if n <= 0:
    return pd.DataFrame()

  faker = Faker()
  Faker.seed(seed)
  np.random.seed(seed)
  hc = Healthcare(generator=faker, dataset_length=n, start_id = 1000001 * (seed+1))

  pd.DataFrame().to_csv(f'file-{seed}.csv', index=False)
  first=True
  for b in hc:
     df = pd.DataFrame(b)
     print(df.shape)
     df.to_csv(
        f'file-{seed}.csv',
        mode='w' if first else 'a', header=first
     )
     first=False

  return None

df_length = 1000233
num_chunks = int(os.cpu_count()) * 2

df_chunks = [df_length // num_chunks] * num_chunks
df_chunks[-1] = df_length - sum(df_chunks[:-1])
Parallel(n_jobs=-1)(delayed(generate_data)(rows, seed) for seed, rows in enumerate(df_chunks))

df_list_pr = list()
for i, _ in enumerate(df_chunks):
   df = pd.read_csv(f'file-{i}.csv')
   df_list_pr.append(df)

df_concat = pd.concat(df_list_pr, ignore_index=True)
end = time.perf_counter()

df_concat.to_csv(
   'updated_synthetic_lung_cancer_dataset_a_2.csv', index=False
)

print(f'parallel: Dataset shape: {df_concat.shape}')
print(f'parallel: Number of collisions: {df_concat['Patient_id'].duplicated().sum()}')
print(f'parallel: Time of execution: {end - start}')

start = time.perf_counter()
faker = Faker()
hc = Healthcare(generator=faker, dataset_length=1000233)

df_list = [pd.DataFrame(b) for b in hc]

df = pd.concat(df_list, ignore_index=True)
end = time.perf_counter()
print(f'iterative: Dataset shape: {df.shape}')
print(f'iterative: Number of collisions: {df['Patient_id'].duplicated().sum()}')
print(f'iterative: Time of execution: {end - start}')

#df.to_csv('updated_synthetic_lung_cancer_dataset_a_1.csv', index=False)