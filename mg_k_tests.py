import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('tables/nutri_comp.csv')
print(df.head().to_string())
nutri_efsa = df.groupby(['efsaprodcode2_recoded', 'NUTRIENT_TEXT']).LEVEL.mean().reset_index()
print(nutri_efsa.head())
nutri_pivot = nutri_efsa.pivot(index='efsaprodcode2_recoded', columns='NUTRIENT_TEXT', values='LEVEL').reset_index()
nutri_pivot.fillna(0, inplace=True)
nutri_pivot = nutri_pivot[['efsaprodcode2_recoded', 'Magnesium (Mg)', 'Potassium (K)', 'Calcium (Ca)']]
print(nutri_pivot.head(10).to_string())

# create columns for normalized values by first copying from original mg and k values
nutri_pivot['mg_norm'] = nutri_pivot['Magnesium (Mg)']
nutri_pivot['k_norm'] = nutri_pivot['Potassium (K)']

# get z-index-normalized values for mg and k
scaler = StandardScaler()
mg_norm = scaler.fit_transform(nutri_pivot.mg_norm.values.reshape(-1, 1))
k_norm = scaler.fit_transform(nutri_pivot.k_norm.values.reshape(-1, 1))

# assign normalized values to columns
nutri_pivot.mg_norm = mg_norm
nutri_pivot.k_norm = k_norm

high_mg = nutri_pivot.sort_values('Magnesium (Mg)', ascending=0).head(100)
high_k = nutri_pivot.sort_values('Potassium (K)', ascending=0).head(100)
low_mg = nutri_pivot.sort_values('Magnesium (Mg)', ascending=1).head(100)
low_k = nutri_pivot.sort_values('Potassium (K)', ascending=1).head(100)

print('High Mg, Low Ca:\n', high_mg.sort_values('Calcium (Ca)').head(20).to_string(), '\n')
print('High K, Low Ca:\n', high_k.sort_values('Calcium (Ca)').head(20).to_string(), '\n')
print('Low Mg, Low Ca:\n', low_mg.sort_values('Calcium (Ca)').head(20).to_string(), '\n')
print('Low K, Low Ca:\n', low_k.sort_values('Calcium (Ca)').head(20).to_string(), '\n')

print(nutri_efsa.NUTRIENT_TEXT.unique())
print('Magnesium Mean:  ', nutri_pivot['Magnesium (Mg)'].mean())
print('Potassium Mean:  ', nutri_pivot['Potassium (K)'].mean())
print('Calcium Mean:    ', nutri_pivot['Calcium (Ca)'].mean())
