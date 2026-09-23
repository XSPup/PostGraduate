# STEP1: add environment
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

print("STEP1: add environment")
print("py executable:", sys.executable)
print("current working directory:", Path.cwd())
print("script path:", Path(__file__).resolve())

# STEP2: add path
print("STEP2: add path")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "kaggle_house"

print("project root:", PROJECT_ROOT)
print("data directory:", DATA_DIR)

# STEP3: check data directory
print("STEP3: check data directory")

if DATA_DIR.exists():
    print("data directory exists")
else:
    DATA_DIR.mkdir(parents=True)
    print("data directory created")

# STEP4: check data files
print("STEP4: check csv files")
TRAIN_CSV_PATH = DATA_DIR / "train.csv"
TEST_CSV_PATH = DATA_DIR / "test.csv"

print("train csv path:", TRAIN_CSV_PATH)
print("test csv path:", TEST_CSV_PATH)

print("train csv exists:", TRAIN_CSV_PATH.exists())
print("test csv exists:", TEST_CSV_PATH.exists())

# STEP5: download csv files if needed
print("STEP5: download csv files if needed")

DATA_HUB = {
    "train": (
        "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv",
        "585e9cc93e70b39160e7921475f9bcd7d31219ce",
        TRAIN_CSV_PATH,
    ),
    "test": (
        "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv",
        "fa19780a7b011d9b009e8bff8e99922a8ee2eb90",
        TEST_CSV_PATH,
    ),
}


def sha1sum(file_path):
    """Compute SHA-1 hash of the file."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(1048576)  # Read in 1 MB chunks
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def download_file(name, url, sha1_hash, save_path):
    if save_path.exists():
        if sha1sum(save_path) == sha1_hash:
            print(f"{name} csv already exists and is valid.")
            return

        print(f"{name} csv exists but sha1 is wrong. Downloading again.")

    print(f"Downloading {name} csv from {url}")
    urllib.request.urlretrieve(url, save_path)

    if sha1sum(save_path) != sha1_hash:
        raise RuntimeError(f"{name} csv downloaded, but sha1 is wrong.")

    print(f"{name} csv downloaded successfully.")


for name, (url, sha1_hash, save_path) in DATA_HUB.items():
    download_file(name, url, sha1_hash, save_path)

# STEP6: load csv files
print("STEP6: load csv files")
train_data = pd.read_csv(TRAIN_CSV_PATH)
test_data = pd.read_csv(TEST_CSV_PATH)

print("train data shape:", train_data.shape)
print("test data shape:", test_data.shape)

print("train data columns:", train_data.columns)
print("first 5 rows of train data:\n", train_data.head(), "\n")

# STEP7: split data into features and labels
print("STEP7: split data into features and labels")

train_features_raw = train_data.drop(columns=["Id", "SalePrice"])
test_features_raw = test_data.drop(columns=["Id"])
train_labels = train_data["SalePrice"]

print("train features shape:", train_features_raw.shape)
print("test features shape:", test_features_raw.shape)
print("train labels shape:", train_labels.shape, "\n")

# STEP8: combine train and test features for preprocessing
print("STEP8: combine train and test features")

all_features_raw = pd.concat([train_features_raw, test_features_raw], axis=0)

print("all features shape:", all_features_raw.shape, "\n")

# STEP9: find numeric features
print("STEP9: find numeric features")

numeric_features = all_features_raw.dtypes[all_features_raw.dtypes != "object"].index

print("number of numeric features:", len(numeric_features))
print("numeric features:")
print(numeric_features)

# STEP10: normalize numeric features
print("STEP10: normalize numeric features")
all_features = all_features_raw.copy()
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
print("first 5 rows of standardized numeric features:")
print(all_features[numeric_features].head())

all_features[numeric_features] = all_features[numeric_features].fillna(0)
print("missing values in numeric features:", all_features[numeric_features].isna().sum().sum())

# STEP11: one-hot encode categorical features
print("STEP11: one-hot encode categorical features")
all_features = pd.get_dummies(all_features, dummy_na=True)
print("all features shape after one-hot encoding:", all_features.shape)
print("remaining missing values:", all_features.isna().sum().sum())

# STEP12: split processed features back into train and test sets
print("\nSTEP12: split processed features")
number_of_train_rows = train_data.shape[0]
