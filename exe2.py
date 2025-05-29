import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import kaggle

dataset_quality = "marcelobatalhah/quality-of-life-index-by-country"
dataset_life = "ignacioazua/life-expectancy"

kaggle.api.dataset_download_files(dataset_quality, path="./", unzip=True)
kaggle.api.dataset_download_files(dataset_life, path="./", unzip=True)

print(os.listdir("./"))

df_quality = pd.read_csv("quality_of_life_indices_by_country.csv")

df_life_expectancy = pd.read_csv("life_expectancy.csv")

print(df_quality.columns)
print(df_life_expectancy.columns)

print(df_quality.head())
print(df_life_expectancy.head())

print(df_quality.info())
print(df_life_expectancy.info())

print(df_quality.describe())
print(df_life_expectancy.describe())

print(df_quality.isnull().sum())
print(df_life_expectancy.isnull().sum())

merged_df = pd.merge(df_quality, df_life_expectancy, on="Country", how="inner")

print(merged_df.describe())
print(merged_df.columns)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x="Sum of Life Expectancy  (both sexes)", y="Quality of Life Index", hue="Country")
plt.title("Relação entre Expectativa de Vida e Qualidade de Vida")
plt.xlabel("Expectativa de Vida (Ambos os sexos)")
plt.ylabel("Índice de Qualidade de Vida")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


plt.figure(figsize=(20, 20))
sns.boxplot(data=merged_df, x="Country", y="Quality of Life Index")
plt.xticks(rotation=90)
plt.title("Distribuição da Qualidade de Vida por País")
plt.xlabel("País")
plt.ylabel("Índice de Qualidade de Vida")
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(merged_df["Sum of Life Expectancy  (both sexes)"], kde=True, color="skyblue")
plt.title("Distribuição da Expectativa de Vida (Ambos os sexos)")
plt.xlabel("Expectativa de Vida")
plt.ylabel("Frequência")
plt.show()


plt.figure(figsize=(20, 20))
sns.barplot(data=merged_df, x="Country", y="Safety Index")
plt.xticks(rotation=90)
plt.title("Índice de Segurança por País")
plt.xlabel("País")
plt.ylabel("Índice de Segurança")
plt.show()


plt.figure(figsize=(20, 20))
sns.barplot(data=merged_df, x="Country", y="Cost of Living Index")
plt.xticks(rotation=90)
plt.title("Índice de Custo de vida por País")
plt.xlabel("País")
plt.ylabel("Índice de Custo de vida")
plt.show()
