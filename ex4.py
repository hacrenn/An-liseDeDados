import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kaggle
import os

dataset_quality = "marcelobatalhah/quality-of-life-index-by-country"
dataset_life    = "ignacioazua/life-expectancy"

kaggle.api.dataset_download_files(dataset_quality, path="./", unzip=True)
kaggle.api.dataset_download_files(dataset_life,    path="./", unzip=True)

df_quality = pd.read_csv("quality_of_life_indices_by_country.csv")
df_life    = pd.read_csv("life_expectancy.csv")


df = pd.merge(df_quality, df_life, on="Country", how="inner")


print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())


# Frequência absoluta de cada ano
year_abs_freq = df["Year"].value_counts().sort_index()
print("=== Frequência absoluta por Year ===")
print(year_abs_freq, "\n")

# Frequência relativa (%) de cada ano
year_rel_freq = df["Year"].value_counts(normalize=True).sort_index() * 100
print("=== Frequência relativa (%) por Year ===")
print(year_rel_freq, "\n")

# Frequência acumulada de cada ano
year_cum_freq = year_abs_freq.cumsum()
print("=== Frequência acumulada por Year ===")
print(year_cum_freq, "\n")

#Gráfico de pizza para distribuição de “Year”
plt.figure(figsize=(5, 5))
year_abs_freq.plot.pie(
    autopct="%1.1f%%",
    startangle=90,
    color="steelblue"
)
plt.title("Distribuição de registros por Year")
plt.ylabel("")  # Remove o label do eixo y
plt.show()

# 6. Gráfico de barras para “Country” com maior Quality of Life
top_countries_qol = df.sort_values(by="Quality of Life Index", ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_countries_qol,
    x="Quality of Life Index",
    y="Country",
    color="steelblue"
)
plt.title("Top 10 Países por Quality of Life Index")
plt.xlabel("Quality of Life Index")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

df["Climate Index"] = pd.to_numeric(df["Climate Index"], errors="coerce")
#Estatísticas descritivas e gráficos para colunas numéricas selecionadas
numeric_cols = [
    "Quality of Life Index",
    "Purchasing Power Index",
    "Safety Index",
    "Health Care Index",
    "Cost of Living Index",
    "Property Price to Income Ratio",
    "Traffic Commute Time Index",
    "Pollution Index",
    "Climate Index",
    "Sum of Life Expectancy  (both sexes)"
]

for col in numeric_cols:
    print(f"\n--- Estatísticas descritivas para '{col}' ---")
    print(f"Média:             {df[col].mean():.2f}")
    print(f"Mediana:           {df[col].median():.2f}")
    print(f"Moda:              {df[col].mode()[0]:.2f}")
    print(f"Amplitude (Range): {df[col].max() - df[col].min():.2f}")
    print(f"Variância:         {df[col].var():.2f}")
    print(f"Desvio Padrão:     {df[col].std():.2f}")
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    print(f"IQR (Q3 – Q1):     {Q3 - Q1:.2f}")

    # Histograma
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"Histograma de '{col}'")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=df[col], color="lightblue")
    plt.title(f"Boxplot de '{col}'")
    plt.tight_layout()
    plt.show()

#Comparação entre média e mediana de “Quality of Life Index”
print("\n--- Mean vs. Median: Quality of Life Index ---")
print(f"Média Quality of Life Index:  {df['Quality of Life Index'].mean():.2f}")
print(f"Mediana Quality of Life Index: {df['Quality of Life Index'].median():.2f}")


#Estatísticas do “Sum of Life Expectancy (both sexes)” agrupadas por Year
le_by_year = df.groupby("Year")["Sum of Life Expectancy  (both sexes)"].describe()
print("\n=== Estatísticas de Expectativa de Vida por Year ===")
print(le_by_year)

# Boxplot de “Sum of Life Expectancy (both sexes)” por Year
plt.figure(figsize=(8, 6))
sns.boxplot(
    x="Year",
    y="Sum of Life Expectancy  (both sexes)",
    data=df,
    color="steelblue"
)
plt.title("Distribuição de Expectativa de Vida por Year")
plt.xlabel("Year")
plt.ylabel("Sum of Life Expectancy (both sexes)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

#Correlação e scatter plot: Quality of Life Index vs Sum of Life Expectancy
corr_qol_le = df[["Quality of Life Index", "Sum of Life Expectancy  (both sexes)"]].corr().iloc[0, 1]
print(f"\nCorrelação entre Quality of Life Index e Expectativa de Vida: {corr_qol_le:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Sum of Life Expectancy  (both sexes)",
    y="Quality of Life Index",
    hue="Year",
    color="steelblue",
    s=50
)
plt.title("Scatter: Qualidade de Vida vs Expectativa de Vida")
plt.xlabel("Sum of Life Expectancy (both sexes)")
plt.ylabel("Quality of Life Index")
plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
