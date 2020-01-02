import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('tables/nutri_comp.csv')

print(df.columns)
df.fillna(0, inplace=True)
df = df[df.UNIT == 'Milligram/100 gram']
df_grouped = df.groupby((['efsaprodcode2_recoded', 'NUTRIENT_TEXT'])).LEVEL.mean().reset_index()
df_pivot = df_grouped.pivot(index='efsaprodcode2_recoded', columns='NUTRIENT_TEXT', values='LEVEL')
print(df_pivot)

plt.scatter(df_pivot['Magnesium (Mg)'], df_pivot['Potassium (K)'], alpha=0.4)
plt.xlim(0, 100)
plt.ylim(0, 1000)
plt.savefig('efsa_mg_k.png')
