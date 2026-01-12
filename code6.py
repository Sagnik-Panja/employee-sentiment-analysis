'''6. Predictive Modeling
Objective

Predict monthly sentiment scores using linear regression.

Selected Features

Message count per month

Average message length

Word count

Feature Engineering'''
df["msg_length"] = df["message"].str.len()
df["word_count"] = df["message"].str.split().apply(len)

features = (
    df.groupby(["employee_id", "month"])
      .agg(
          msg_count=("message", "count"),
          avg_length=("msg_length", "mean"),
          avg_words=("word_count", "mean"),
          sentiment_score=("sentiment_score", "sum")
      )
      .reset_index()
)
#Model Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = features[["msg_count", "avg_length", "avg_words"]]
y = features["sentiment_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
'''Interpretation

Higher message frequency often correlates with stronger sentiment (positive or negative).

Longer messages are more likely emotionally charged.

Model provides directional insights, not exact prediction guarantees.'''