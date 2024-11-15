import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

df = pd.read_csv('/content/data.csv')
print("First few rows of the data:")
print(df.head())


print("\nMissing values:") #check for missing values
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median()) #handling
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)


print("\nMissing values after handling:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# 1. Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived', palette='Set2')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# 2. Survival Count by Gender
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', hue='Sex', palette='Set1')
plt.title("Survival Count by Gender")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Gender")
plt.show()

# 3. Survival Count by Passenger Class
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived', hue='Pclass', palette='Set3')
plt.title("Survival Count by Passenger Class")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Passenger Class")
plt.show()

# 4. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 5. Survival Count by Age Group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], 
                        labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='AgeGroup', hue='Survived', palette='muted')
plt.title("Survival Count by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="Survived")
plt.show()

# 6. Fare Distribution by Passenger Class and Survival
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Survived', palette='coolwarm')
plt.title("Fare Distribution by Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

# Generate Profile Report
profile = ProfileReport(df, title="Train Data Profiling Report", explorative=True)
profile.to_file("data.html")
