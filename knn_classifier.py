from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd

df = pd.read_csv('tables/nutri_comp.csv')
print(df.columns)
df = df.groupby(['NUTRIENT_TEXT', 'level3']).LEVEL.mean().reset_index()
nutrients = df.NUTRIENT_TEXT.unique().tolist()
nutri_tab = df.pivot(index='level3', columns='NUTRIENT_TEXT', values='LEVEL').reset_index()
nutri_tab.fillna(0, inplace=True)
print(nutri_tab.columns)

scale = StandardScaler()

for x in nutri_tab[nutrients]:
    nutri_tab[x] = scale.fit_transform(nutri_tab[x].values.reshape(-1, 1))

feature_train, feature_test, label_train, label_test = train_test_split(nutri_tab[nutrients], nutri_tab.level3,
                                                                        train_size=0.8)
model = KNeighborsClassifier(n_neighbors=3, weights='distance')
model.fit(feature_train, label_train)
print(model.score(feature_train, label_train))
y_predict = model.predict(feature_test)
print(y_predict)
print(model.score(feature_test, label_test))

y_predict = y_predict.tolist()
label_test = label_test.tolist()

