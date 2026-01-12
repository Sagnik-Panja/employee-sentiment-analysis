'''4. Employee Ranking
Objective

Rank employees by monthly sentiment scores.

Requirements Implemented

Top 3 Positive employees

Top 3 Negative employees

Sorted by score, then alphabetically

Code'''
def rank_employees(month):
    data = monthly_scores[monthly_scores["month"] == month]

    top_positive = (
        data.sort_values(["sentiment_score", "employee_id"],
                          ascending=[False, True])
            .head(3)
    )

    top_negative = (
        data.sort_values(["sentiment_score", "employee_id"],
                          ascending=[True, True])
            .head(3)
    )

    return top_positive, top_negative
