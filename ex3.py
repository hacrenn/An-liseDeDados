import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import kaggle

# Load the dataset
file_path = "/Users/vascofelgueiras/Desktop/Universidade/AnaliseDados/Diabetes Missing Data.csv"
df = pd.read_csv(file_path)

print("Initial Data Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values per Column:")
print(df.isnull().sum())

missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

#Handling Missing Values
for col in ["Glucose", "Diastolic_BP", "Skin_Fold", "Serum_Insulin", "BMI"]: df[col].fillna(df[col].mean(), inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

#Boxplot Before Handling Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["Glucose", "Diastolic_BP", "Skin_Fold", "Serum_Insulin", "BMI"]])
plt.title("Boxplot Before Outlier Handling")
plt.show()

#Outliers specifically
Q1 = df["BMI"].quantile(0.25)
Q3 = df["BMI"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["BMI"] < lower_bound) | (df["BMI"] > upper_bound)]
print(outliers)

#Boxplot Before Handling Outliers BMI
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["BMI"])
plt.title("Boxplot Before Outlier Handling BMI")
plt.show()

#Handling Outliers
#Outliers to low or upper bound
for col in ["Glucose", "Diastolic_BP", "Skin_Fold", "Serum_Insulin", "BMI"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("\nDataset After Handling Outliers:")
print(df.describe())

#Boxplot After Handling Outliers BMI
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["BMI"])
plt.title("Boxplot After Outlier Handling BMI")
plt.show()

#Save
df.to_csv("/Users/vascofelgueiras/Desktop/Universidade/AnaliseDados/Diabetes Missing Data.csv", index=False)
print("Cleaned dataset saved as 'Diabetes Missing Data.csv'")
