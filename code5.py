'''5. Flight Risk Identification
Objective

Identify employees at flight risk.

Definition

An employee is at risk if they send 4 or more negative messages within any rolling 30-day window.

Code'''
df = df.sort_values(["employee_id", "date"])

df["negative_flag"] = (df["sentiment_label"] == "Negative").astype(int)

df["neg_30d"] = (
    df.groupby("employee_id")["negative_flag"]
      .rolling("30D", on=df["date"])
      .sum()
      .reset_index(level=0, drop=True)
)

flight_risk_employees = df[df["neg_30d"] >= 4]["employee_id"].unique()
