'''3. Employee Score Calculation
Objective

Compute monthly sentiment scores per employee.

Scoring Rules
Sentiment	Score
Positive	+1
Neutral	    0
Negative	−1'''
#Code
# Convert to datetime
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M")

score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
df["sentiment_score"] = df["sentiment_label"].map(score_map)

monthly_scores = (
    df.groupby(["employee_id", "month"])["sentiment_score"]
      .sum()
      .reset_index()
)
