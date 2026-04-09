print("RUNNING SCRIPT")

import pandas as pd
import xgboost as xgb
import duckdb

# -----------------------------
# Load current season data
# -----------------------------
curr = pd.read_csv("Data/current_season.csv")

print("Columns:")
print(curr.columns.tolist())

# -----------------------------
# Basic cleaning
# -----------------------------
percent_cols = ["FG%", "2P%", "3P%", "FT%", "TS%", "eFG%"]
for col in percent_cols:
    if col in curr.columns:
        curr[col] = curr[col].fillna(0)

curr = curr.replace([float("inf"), float("-inf")], 0)
curr = curr.fillna(0)

# -----------------------------
# Features that MUST match training
# -----------------------------
feature_cols = [
    "GP", "GS", "MP", "FG", "FGA", "2P", "2PA", "3P", "3PA", "FT", "FTA",
    "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "FG%", "2P%", "3P%", "FT%", "TS%", "eFG%"
]

missing = [col for col in feature_cols if col not in curr.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# -----------------------------
# Load trained model
# -----------------------------
model = xgb.XGBClassifier()
model.load_model("xgb_final.json")

# -----------------------------
# Predict draft probabilities
# -----------------------------
curr["Draft_Prob"] = model.predict_proba(curr[feature_cols])[:, 1]

# -----------------------------
# Create per-game stats for dashboard
# -----------------------------
curr["PPG"] = curr["PTS"] / curr["GP"].replace(0, 1)
curr["APG"] = curr["AST"] / curr["GP"].replace(0, 1)
curr["RPG"] = curr["TRB"] / curr["GP"].replace(0, 1)
# -----------------------------
# Print top 10 players
# -----------------------------
top10 = curr.sort_values("Draft_Prob", ascending=False).head(10)

print("\nTop 10 players by draft probability:")
print(top10[["Player", "Team", "Draft_Prob"]])

# -----------------------------
# Print favorite team players
# Change team name if needed
# -----------------------------
favorite_team = "Michigan State"

team_df = curr[curr["Team"] == favorite_team].sort_values("Draft_Prob", ascending=False)

print(f"\nPlayers on {favorite_team}:")
if len(team_df) > 0:
    print(team_df[["Player", "Team", "Draft_Prob"]])
else:
    print("No players found for that team. Check the exact team name in the CSV.")

# -----------------------------
# Save to DuckDB
# -----------------------------
con = duckdb.connect("curr_season.duckdb")
con.execute("DROP TABLE IF EXISTS curr")
con.execute("CREATE TABLE curr AS SELECT * FROM curr")
con.close()

print("\nSaved predictions to curr_season.duckdb")

# -----------------------------
# Verify with SQL
# -----------------------------
con = duckdb.connect("curr_season.duckdb")

result = con.execute("""
    SELECT Player, Team, Draft_Prob
    FROM curr
    ORDER BY Draft_Prob DESC
    LIMIT 10
""").fetchdf()

print("\nTop 10 from DuckDB:")
print(result)

con.close()
con = duckdb.connect("curr_season.duckdb")

msu_result = con.execute("""
    SELECT Player, Team, Draft_Prob
    FROM curr
    WHERE Team = 'Michigan State'
    ORDER BY Draft_Prob DESC
""").fetchdf()

print("\nMichigan State players from DuckDB:")
print(msu_result)

con.close()