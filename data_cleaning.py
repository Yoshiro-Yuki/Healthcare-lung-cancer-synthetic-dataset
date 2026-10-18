"""Data cleaner for the messy synthetic lung-cancer dataset.

This module defines ``LungCancerCleaner``, a class that takes the messy
dataset produced by ``DirtyDataInjector`` and returns a cleaned *copy*. It
reproduces the steps of the original ``cleaning_lung_cancer_v2`` notebook:

    1. Numeric columns: text and negative ages become missing, then missing
       values are filled with the column median.
    2. Categorical columns: whitespace and casing are fixed, nulls become
       ``Unknown``, and typo variants are mapped to one standard label.
    3. Binary columns: 0 and 1 become ``No`` and ``Yes``, and missing values
       are filled with the most common answer.
    4. Dates and types: Visit_date is parsed and label columns become
       categories.

The cleaner never reads or writes files and never edits the DataFrame it is
given, so it can be used on a batch, a full DataFrame or a Parquet file.

All data is entirely synthetic and must not be used for clinical decisions.

Typical usage example:

    messy_df = pd.read_parquet('messy_data.parquet')
    cleaner = LungCancerCleaner(verbose=True)
    clean_df = cleaner.clean(messy_df)
    clean_df.to_parquet('cleaned_data_v2.parquet', index=False)
"""

# --- Imports ---------------------------------------------------------------
import numpy as np                            # NaN for invalid ages
import pandas as pd                           # DataFrame cleaning


class LungCancerCleaner:
  """Cleans a messy lung-cancer DataFrame step by step.

  The cleaner is stateful during a call: ``clean`` copies the input into
  ``df``, each cleaning step edits ``df`` in place, and the finished copy is
  returned. The steps depend on each other, so their order matters. For
  example, binary columns must be mapped to ``Yes``/``No`` before their
  missing values can be filled.

  Attributes:
    DROP_COLUMNS: Identifier and index columns removed from the output.
    NUMERIC_COLUMNS: Columns converted to numbers and filled with the median.
    CATEGORICAL_COLUMNS: Text columns that are stripped, title-cased and
      filled with ``Unknown``.
    BINARY_COLUMNS: 0/1 columns converted to ``No``/``Yes``.
    CATEGORY_COLUMNS: Columns cast to the pandas ``category`` type at the end
      of the typing step.
    BINARY_LABELS: Mapping from 0/1 to ``No``/``Yes``.
    GENDER_MAP: Lower-case Gender variants mapped to a standard label.
    SMOKING_MAP: Lower-case Smoking_status variants mapped to a standard
      label.
    MISPLACED_GENDER_VALUES: Gender-like values found in Smoking_status.
    NEVER_ALIASES: Smoking_status variants that all mean ``Never``.
    verbose: If True, the number of missing values is printed after each step.
    df: The working copy of the DataFrame (``None`` until ``clean`` runs).
  """

  DROP_COLUMNS = ['Patient_id', 'Record_id', 'branch_id']

  NUMERIC_COLUMNS = ['Age', 'Pack_years']

  CATEGORICAL_COLUMNS = [
    'Gender', 'Smoking_status', 'Air_pollution_exposure', 'Genetic_risk',
  ]

  BINARY_COLUMNS = [
    'Chronic_lung_disease',
    'Chest_pain',
    'Shortness_of_breath',
  ]

  CATEGORY_COLUMNS = [
    'Chronic_lung_disease',
    'Chest_pain',
    'Shortness_of_breath',
    'Diagnosis_result',
  ]

  BINARY_LABELS = {0: 'No', 1: 'Yes'}

  GENDER_MAP = {
    'male': 'Male',
    'm': 'Male',
    'masc': 'Male',
    'female': 'Female',
    'fem': 'Female',
    'f': 'Female',
    'unknown': 'Unknown',
  }

  SMOKING_MAP = {
    'current': 'Current',
    'cur': 'Current',
    'now': 'Current',
    'former': 'Former',
    'form3r': 'Former',
    'before': 'Former',
    'never': 'Never',
    'nev': 'Never',
    'nvever': 'Never',
    'unknown': 'Unknown',
  }

  MISPLACED_GENDER_VALUES = ['male', 'masc', 'm']
  NEVER_ALIASES = ['negative', 'no', 'didnt', 'none']

  def __init__(self, verbose=False):
    """Initializes the cleaner.

    Args:
      verbose: If True, print the total number of missing values after each
        cleaning step.
    """
    self.verbose = verbose
    self.df = None                            # Set by clean()

  def clean(self, dataframe):
    """Runs every cleaning step on a copy of ``dataframe``.

    Args:
      dataframe: A messy DataFrame. It is not modified.

    Returns:
      A new, cleaned DataFrame.
    """
    self.df = dataframe.copy()

    (
      self.drop_columns()
      .clean_numeric()
      .impute_numeric()
      .clean_categorical()
      .encode_binary()
      .fill_binary_unknown()
      .parse_dates()
      .cast_categories()
      .standardize_labels()
      .fix_misplaced_smoking_values()
      .fix_never_aliases()
      .impute_binary()
    )

    return self.df

  def missing_report(self):
    """Counts the missing values in every column.

    Returns:
      A DataFrame with the count and the percentage of missing values per
      column.
    """
    counts = self.df.isna().sum()
    return pd.DataFrame({
      'missing': counts,
      'percent': counts * 100 / len(self.df),
    })

  def _log(self, step):
    """Prints the total number of missing values when ``verbose`` is on.

    Args:
      step: Name of the step that just finished.
    """
    if self.verbose:
      print(f'{step}: {int(self.df.isna().sum().sum())} missing values')

  def drop_columns(self):
    """Removes identifier and index columns, ignoring any that are absent.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    #self.df = self.df.drop(columns=self.DROP_COLUMNS, errors='ignore')
    self._log('drop_columns')
    return self

  def clean_numeric(self):
    """Converts Age and Pack_years to numbers and removes negative ages.

    Text such as ``Twenty Five`` cannot be converted and becomes missing.
    Only negative ages are removed here; other impossible values (for
    example an age of 394, or a negative Pack_years) are kept.

    Returns:
      The cleaner itself, so calls can be chained.
    """

    for column in self.NUMERIC_COLUMNS:
      self.df[column] = pd.to_numeric(self.df[column], errors='coerce')


    self.df.loc[(self.df['Age'] > 120) | (self.df['Age'] < 0), 'Age'] = np.nan
    self._log('clean_numeric')
    return self

  def impute_numeric(self):
    """Fills missing Age and Pack_years values with each column's median.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    medians = self.df[self.NUMERIC_COLUMNS].median()
    self.df[self.NUMERIC_COLUMNS] = self.df[self.NUMERIC_COLUMNS].fillna(medians)
    self._log('impute_numeric')
    return self

  def clean_categorical(self):
    """Strips whitespace, title-cases text and fills nulls with ``Unknown``.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    for column in self.CATEGORICAL_COLUMNS:
      self.df[column] = (
        self.df[column]
        .astype('string')
        .str.strip()
        .str.title()
        .fillna('Unknown')
      )

    self._log('clean_categorical')
    return self

  def encode_binary(self):
    """Maps the 0/1 columns to ``No``/``Yes``.

    The values are converted to numbers first, so the step also works when the
    data was read from a file where every column is text (``"0"`` and
    ``"1"``). Anything that is not 0 or 1 becomes missing.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    for column in self.BINARY_COLUMNS:
      numbers = pd.to_numeric(self.df[column], errors='coerce')
      self.df[column] = numbers.map(self.BINARY_LABELS)

    self._log('encode_binary')
    return self

  def fill_binary_unknown(self):
    """Fills missing values in the binary columns with ``Unknown``.

    ``impute_binary`` later replaces these with the most common answer.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    for column in self.BINARY_COLUMNS:
      self.df[column] = self.df[column].fillna('Unknown')

    self._log('fill_binary_unknown')
    return self

  def parse_dates(self):
    """Parses Visit_date, reading day first. Invalid dates become missing.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    self.df['Visit_date'] = pd.to_datetime(
      self.df['Visit_date'],
      format='mixed',
      dayfirst=True,
      errors='coerce',
    )

    # Removes visits after 2026
    self.df = self.df[~(self.df['Visit_date'].dt.year > 2026)]

    self._log('parse_dates')
    return self

  def cast_categories(self):
    """Casts the label columns to the pandas ``category`` type.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    for column in self.CATEGORY_COLUMNS:
      self.df[column] = self.df[column].astype('category')

    self._log('cast_categories')
    return self

  @staticmethod
  def _standardize_values(series, mapping):
    """Maps typo variants to a standard label.

    Values are stripped and lower-cased before the lookup. Values that are not
    in the mapping are kept in title case.

    Args:
      series: The column to standardize.
      mapping: Dict from lower-case variant to standard label.

    Returns:
      A new Series with standardized labels.
    """
    cleaned = series.astype('string').str.strip().str.casefold()
    return cleaned.map(mapping).fillna(cleaned.str.title())

  def standardize_labels(self):
    """Maps Gender and Smoking_status typos to standard labels.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    self.df['Gender'] = self._standardize_values(
      self.df['Gender'], self.GENDER_MAP
    )
    self.df['Smoking_status'] = self._standardize_values(
      self.df['Smoking_status'], self.SMOKING_MAP
    )

    self._log('standardize_labels')
    return self

  def _smoking_values(self):
    """Returns Smoking_status as stripped, lower-case text for matching."""
    return (
      self.df['Smoking_status']
      .astype('string')
      .str.strip()
      .str.casefold()
    )

  def fix_misplaced_smoking_values(self):
    """Sets gender-like values found in Smoking_status to ``Unknown``.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    misplaced = self._smoking_values().isin(self.MISPLACED_GENDER_VALUES)
    self.df.loc[misplaced, 'Smoking_status'] = 'Unknown'

    self._log('fix_misplaced_smoking_values')
    return self

  def fix_never_aliases(self):
    """Maps ``Negative``, ``No``, ``Didnt`` and ``None`` to ``Never``.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    never = self._smoking_values().isin(self.NEVER_ALIASES)
    self.df.loc[never, 'Smoking_status'] = 'Never'

    self._log('fix_never_aliases')
    return self

  def impute_binary(self):
    """Replaces missing and ``Unknown`` binary answers with the most common one.

    The most common answer is calculated from the known answers only. A column
    with no known answers is left unchanged.

    Returns:
      The cleaner itself, so calls can be chained.
    """
    for column in self.BINARY_COLUMNS:
      values = self.df[column].astype('string')
      unknown = values.isna() | values.str.casefold().eq('unknown')
      known = values.loc[~unknown]

      if not known.empty:
        self.df.loc[unknown, column] = known.mode().iloc[0]

    self._log('impute_binary')
    return self