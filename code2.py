'''2. Exploratory Data Analysis (EDA)
Objectives

Understand dataset structure

Analyze sentiment distribution

Identify temporal trends and anomalies

Key EDA Steps'''
# Basic structure
df.info()
df.describe(include="all")

# Missing values
df.isnull().sum()

# Sentiment distribution
df["sentiment_label"].value_counts()
#Visualizations
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="sentiment_label", data=df)
plt.title("Distribution of Sentiment Labels")
plt.show()
'''Insights from EDA

Neutral messages dominate typical corporate communication.

Negative spikes often align with deadlines or workload peaks.

Certain employees show persistent sentiment trends.'''