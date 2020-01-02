from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

df = pd.read_csv('tables/nutri_comp.csv')
df = df.groupby(['efsaprodcode2_recoded', 'NUTRIENT_TEXT']).LEVEL.mean().reset_index()
df = df.pivot(index='efsaprodcode2_recoded', columns='NUTRIENT_TEXT', values='LEVEL').reset_index()
df.fillna(0, inplace=True)
data = df.drop('efsaprodcode2_recoded', axis=1)
target = df['efsaprodcode2_recoded']
print(data.columns)
print(target.head())


training_set, validation_set = train_test_split(data, target, train_size=0.8)