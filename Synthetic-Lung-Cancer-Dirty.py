import numpy as np
import pandas as pd


df = pd.read_csv("updated_synthetic_lung_cancer_dataset_a_2.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

rng = np.random.default_rng(67)
row_count = len(df)

dates = [
    "10-10-2000", "19-12-2006", "October 8, 2024",
    "03-04-2001", "15-08-2003", "22-11-2005", "07-01-2008", "30-06-2010",
    "12-09-2012", "25-12-2014", "01-03-2015",
    "14-02-2028", "09-07-2029", "31-12-2030",
    "2024/13/45", "Oct 5th, 2024", "5 October 2024", "2024.10.05",
    "10/05/24", "05-Oct-24", "yesterday", "N/A", "00-00-0000",
    "Oct-2024", "2024-10-5", "October 32, 2024", "12th of Dec 2023",
    "sometime in 2022", "20241005", "  2024-10-05  ", "10--10--2024",
]

null_columns = [
    "Age",
    "Gender",
    "Smoking_status",
    "Pack_years",
    "Air_pollution_exposure",
    "Genetic_risk",
    "Chest_pain",
    "Chronic_lung_disease",
    "Shortness_of_breath",
]

MIN_NULL_RATE = 0.0263
MAX_NULL_RATE = 0.0863

for column in null_columns:
    null_rate = rng.uniform(MIN_NULL_RATE, MAX_NULL_RATE)
    is_null = rng.random(row_count) < null_rate
    df.loc[is_null, column] = pd.NA


female_mask = (
    df["Gender"].eq("Female")
    & (rng.random(row_count) < 0.02543)
)

male_mask = (
    df["Gender"].eq("Male")
    & (rng.random(row_count) < 0.0253)
)

Status_maskF = (
    df["Smoking_status"].eq("Current")
    & (rng.random(row_count) < 0.0393)
)

Status_maskN = (
    df["Smoking_status"].eq("Never")
    & (rng.random(row_count) < 0.073)
)

Status_maskC = (
    df["Smoking_status"].eq("Former")
    & (rng.random(row_count) < 0.12)
)

pack_mask = rng.random(row_count) < 0.001
Age_mask = rng.random(row_count) < 0.042
date_mask = rng.random(row_count) < 0.00153

df.loc[female_mask, "Gender"] = rng.choice(
    ["F", "female", "fem", "Female "],
    size=female_mask.sum()
)

df.loc[male_mask, "Gender"] = rng.choice(
    ["M", "male", "masc", "Male "],
    size=male_mask.sum()
)

df.loc[Status_maskF, "Smoking_status"] = rng.choice(
    ["Before", "ForM3R", "former", "FORmER "],
    size=Status_maskF.sum()
)

df.loc[Status_maskC, "Smoking_status"] = rng.choice(
    ["CUR", "Current ", "CURrENT", "NOW "],
    size=Status_maskC.sum()
)

df.loc[Status_maskN, "Smoking_status"] = rng.choice(
    ["didnt", "NEv", "Negative", "None ", "No", 'NEvER'],
    size=Status_maskN.sum()
)


df.loc[pack_mask, "Pack_years"] = rng.choice(
    [-28, 350, "0", "Twenty three", "forty two", 100, 200, 300, 400, "twenty five",],
    size=pack_mask.sum()
)

messy_values = [-15, "SIx seven", "69", "Twenty Five", 394, 900, 132, 242]
messy_values += rng.integers(-60, -22, size=10).tolist()

df.loc[Age_mask, "Age"] = [rng.choice(messy_values) for _ in range(Age_mask.sum())]

df.loc[date_mask, "Visit_date"] = rng.choice(
    dates,
    size=date_mask.sum()
)

df.to_csv("cleaned_lung_cancer_dataset_v2.csv",
    index=False
)