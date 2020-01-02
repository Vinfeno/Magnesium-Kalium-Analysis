import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('tables/nutri_comp.csv')
df = df[df.UNIT == 'Milligram/100 gram']
df = df.fillna(0)
df = df.rename(columns={'Unnamed: 0': 'ID'})
df_mg = df[df.NUTRIENT_TEXT == 'Magnesium (Mg)']
df_k = df[df.NUTRIENT_TEXT == 'Potassium (K)']
df_mg = df_mg.rename(columns={'LEVEL': 'LEVEL_MG'})
df_k = df_k.rename(columns={'LEVEL': 'LEVEL_K'})
print(df)

# df_mg.LEVEL_MG.to_csv('df_mg.csv')
# df_k.LEVEL_K.to_csv('df_k.csv')
# print(df_mg.head().to_string())
# print(df_k.head().to_string())

df_mg_k = pd.read_csv('tables/df_mg_k.csv')
df_mg_k.fillna(0, inplace=True)
df_mg_k.columns = ['LEVEL_MG', 'LEVEL_K']
print(df_mg_k.head(100).to_string())

mg_train, mg_test, k_train, k_test = train_test_split(df_mg_k.LEVEL_MG, df_mg_k.LEVEL_K, train_size=0.8)

mg_train = mg_train.values.reshape(-1, 1)
mg_test = mg_test.values.reshape(-1, 1)
k_train = k_train.values.reshape(-1, 1)
k_test = k_test.values.reshape(-1, 1)

lr = LinearRegression()
model = lr.fit(mg_train, k_train)
y_predict = model.predict(mg_test)

plt.scatter(df_mg_k.LEVEL_MG, df_mg_k.LEVEL_K, alpha=0.4)
plt.xlim(0, 5000)
plt.ylim(0, 600)
plt.savefig('MG_K.png')

# df_pivot = df.pivot(index='FOOD_ID', columns='NUTRIENT_TEXT', values='LEVEL')
# mg_train, mg_test, k_train, k_test = train_test_split(df_mg.LEVEL, df_k.LEVEL)
