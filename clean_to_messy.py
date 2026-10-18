import numpy as np
import pandas as pd


class DirtyDataInjector:
    """Corrupts a clean dataset with nulls, typos and impossible values."""

    MIN_NULL_RATE = 0.0263
    MAX_NULL_RATE = 0.0483

    NULL_COLUMNS = [
        'Age',
        'Gender',
        'Smoking_status',
        'Pack_years',
        'Air_pollution_exposure',
        'Genetic_risk',
        'Chest_pain',
        'Chronic_lung_disease',
        'Shortness_of_breath',
    ]

    MESSY_DATES = [
        '10-10-2000', '19-12-2006', 'October 8, 2024',
        '03-04-2001', '15-08-2003', '22-11-2005', '07-01-2008', '30-06-2010',
        '12-09-2012', '25-12-2014', '01-03-2015',
        '14-02-2028', '09-07-2029', '31-12-2030',
        '2024/13/45', 'Oct 5th, 2024', '5 October 2024', '2024.10.05',
        '10/05/24', '05-Oct-24', 'yesterday', 'N/A', '00-00-0000',
        'Oct-2024', '2024-10-5', 'October 32, 2024', '12th of Dec 2023',
        'sometime in 2022', '20241005', '  2024-10-05  ', '10--10--2024',
    ]

    FEMALE_TYPOS = ['F', 'female', 'fem', 'Female ']
    MALE_TYPOS = ['M', 'male', 'masc', 'Male ']
    FORMER_TYPOS = ['Before', 'ForM3R', 'former', 'FORmER ']
    CURRENT_TYPOS = ['CUR', 'Current ', 'CURrENT', 'NOW ']
    NEVER_TYPOS = ['didnt', 'NEv', 'Negative', 'None ', 'No', 'NEvER']

    MESSY_PACK_YEARS = [
        -28, 350, '0', 'Twenty three', 'forty two', 100, 200, 300, 400,
        'twenty five',
    ]
    MESSY_AGES = [-15, 'SIx seven', '69', 'Twenty Five', 394, 900, 132, 242]

    def __init__(self, seed=67):
        """Initializes the injector with a random seed."""
        self.rng = np.random.default_rng(seed)
        self.df = None
        self.masks = {}

    def corrupt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all corruption steps to a copy of the input DataFrame.

        Args:
            df: Clean pandas DataFrame.

        Returns:
            A new corrupted pandas DataFrame.
        """
        # Work on a copy so the original clean DataFrame is untouched
        self.df = df.copy()

        for col in ['Pack_years', 'Age', 'Visit_date']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(object)

        (
            self.inject_nulls()
            .build_masks()
            .apply_categorical_typos()
            .apply_numeric_noise()
            .apply_date_noise()
        )

        return self.df

    def inject_nulls(self):
        row_count = len(self.df)
        for column in self.NULL_COLUMNS:
            null_rate = self.rng.uniform(self.MIN_NULL_RATE, self.MAX_NULL_RATE)
            is_null = self.rng.random(row_count) < null_rate
            self.df.loc[is_null, column] = pd.NA
        return self

    def build_masks(self):
        row_count = len(self.df)
        rng = self.rng
        self.masks = {
            'female': self.df['Gender'].eq('Female') & (rng.random(row_count) < 0.02543),
            'male': self.df['Gender'].eq('Male') & (rng.random(row_count) < 0.0253),
            'current': self.df['Smoking_status'].eq('Current') & (rng.random(row_count) < 0.0393),
            'never': self.df['Smoking_status'].eq('Never') & (rng.random(row_count) < 0.073),
            'former': self.df['Smoking_status'].eq('Former') & (rng.random(row_count) < 0.12),
            'pack': rng.random(row_count) < 0.001,
            'age': rng.random(row_count) < 0.042,
            'date': rng.random(row_count) < 0.00153,
        }
        return self

    def apply_categorical_typos(self):
        masks = self.masks
        self.df.loc[masks['female'], 'Gender'] = self.rng.choice(
            self.FEMALE_TYPOS, size=masks['female'].sum()
        )
        self.df.loc[masks['male'], 'Gender'] = self.rng.choice(
            self.MALE_TYPOS, size=masks['male'].sum()
        )
        self.df.loc[masks['current'], 'Smoking_status'] = self.rng.choice(
            self.FORMER_TYPOS, size=masks['current'].sum()
        )
        self.df.loc[masks['former'], 'Smoking_status'] = self.rng.choice(
            self.CURRENT_TYPOS, size=masks['former'].sum()
        )
        self.df.loc[masks['never'], 'Smoking_status'] = self.rng.choice(
            self.NEVER_TYPOS, size=masks['never'].sum()
        )
        return self

    def apply_numeric_noise(self):
        masks = self.masks
        self.df.loc[masks['pack'], 'Pack_years'] = self.rng.choice(
            self.MESSY_PACK_YEARS, size=masks['pack'].sum()
        )
        messy_ages = self.MESSY_AGES + self.rng.integers(-60, -22, size=10).tolist()
        self.df.loc[masks['age'], 'Age'] = [
            self.rng.choice(messy_ages) for _ in range(masks['age'].sum())
        ]
        return self

    def apply_date_noise(self):
        self.df.loc[self.masks['date'], 'Visit_date'] = self.rng.choice(
            self.MESSY_DATES, size=self.masks['date'].sum()
        )
        return self