import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Task1: perform data cleaning, aggregation, and filtering
#load titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#inspect data
print(df.info())
print(df.describe())

#handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

#Remove duplicates
df = df.drop_duplicates()

#Filter data: Passengers in first class
first_class = df[df["Pclass"]==1]
print("First Class Passenger:\n",first_class.head())

#Task2: generate visualization to illustrate key insights
#bar chart:Survival rate by class
survival_by_class = df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar", color ="purple")
plt.title("Survival Rate by class")
plt.ylabel("Survival Rate")
plt.show()


#histogram : Age distribution
sns.histplot(df["Age"] ,kde =True, bins=20, color="brown")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Scatterplot: Age vs Fare
plt.scatter(df["Age"],df["Fare"],alpha=0.5, color="darkblue")
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

