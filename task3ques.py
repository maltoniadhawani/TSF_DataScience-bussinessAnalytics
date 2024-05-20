import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

samplestore = pd.read_csv('samplestore.csv')

print(samplestore.head())

print(samplestore.describe())

print(samplestore.info())

sns.pairplot(samplestore)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(samplestore.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Ship Mode', y='Sales', data=samplestore)
plt.title('Sales by Ship Mode')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Profit', data=samplestore)
plt.title('Profit by Region')
plt.show()
