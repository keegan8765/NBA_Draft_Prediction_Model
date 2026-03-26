import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
ncaa = pd.read_csv("Data/ncaa_data.csv")

# Clean columns
ncaa["Class"] = ncaa["Class"].str.strip()

# Fill missing values
ncaa["3P%"] = ncaa["3P%"].fillna(0)

# Feature engineering
ncaa["PPG"] = ncaa["PTS"] / ncaa["GP"]
ncaa["RPG"] = ncaa["TRB"] / ncaa["GP"]
ncaa["APG"] = ncaa["AST"] / ncaa["GP"]
ncaa["MP"] = ncaa["MP"] / ncaa["GP"]
ncaa["STL"] = ncaa["STL"] / ncaa["GP"]
ncaa["BLK"] = ncaa["BLK"] / ncaa["GP"]
ncaa["TOV"] = ncaa["TOV"] / ncaa["GP"]
ncaa["3PAr"] = ncaa["3PA"] / ncaa["FGA"]
ncaa["FTr"] = ncaa["FT"] / ncaa["FG"]
ncaa["GS_rate"] = ncaa["GS"] / ncaa["GP"]

# Encoding
pos_map = {"G": 0, "F": 1, "C": 2}
ncaa["POS_enc"] = ncaa["POS"].map(pos_map)

class_map = {"FR": 0, "SO": 1, "JR": 2, "SR": 3}
ncaa["Class_enc"] = ncaa["Class"].map(class_map)

# Features and target
features = [
    "PPG", "RPG", "APG", "MP", "STL", "BLK",
    "TOV", "3PAr", "FTr", "GS_rate", "POS_enc", "Class_enc"
]

X = ncaa[features]
y = ncaa["Drafted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed datasets
train_df = X_train.copy()
train_df["Drafted"] = y_train

test_df = X_test.copy()
test_df["Drafted"] = y_test

train_df.to_csv("Data/train.csv", index=False)
test_df.to_csv("Data/test.csv", index=False)

print("Preprocessing complete")