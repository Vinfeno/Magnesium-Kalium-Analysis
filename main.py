import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

# read and instantiate csv table
df = pd.read_csv('tables/nutri_comp.csv')

# transform into different tables
grp_rec_lv1 = df.groupby(['efsaprodcode2_recoded', 'level1', 'level2', 'level3']).FOOD_ID.count().reset_index()
grp_rec_lv1 = grp_rec_lv1.sort_values('FOOD_ID', ascending=0)
# print(df.level3.unique())
nutri_grp = df[df.UNIT != 'Microgram/100 gram']
nutri_grp = df.groupby(['level3', 'NUTRIENT_TEXT']).LEVEL.mean().reset_index()
# print(nutri_grp.head(20).to_string())
potassium = nutri_grp[nutri_grp.NUTRIENT_TEXT == 'Potassium (K)']
potassium_high = potassium.sort_values('LEVEL', ascending=0).head(20)
potassium_low = potassium.sort_values('LEVEL').head(20)
# print(potassium.head(10))
magnesium = nutri_grp[nutri_grp.NUTRIENT_TEXT == 'Magnesium (Mg)']
magnesium_high = magnesium.sort_values('LEVEL', ascending=0).head(20)
magnesium_low = magnesium.sort_values('LEVEL').head(20)
# print(magnesium.head(10))
# final pivot table
nutri_pivot = nutri_grp.pivot(index='level3', columns='NUTRIENT_TEXT', values='LEVEL').fillna(0)


# print(nutri_pivot.to_string())

def nutri_dist():
    # generate histogram of all nutrients
    plt.figure(figsize=(10, 10))
    for x in nutri_pivot.columns:
        sns.kdeplot(nutri_pivot[x])
    plt.xlim(left=0, right=0.5)
    plt.title('Distribution of Nutrients')
    plt.xlabel('Milligram/ 100 gram')
    plt.savefig('./plots/nutri_dist.png')
    plt.cla()


def predict():
    # prediction model with linear regression
    k_train, k_test, mg_train, mg_test = train_test_split(potassium.LEVEL.values.reshape(-1, 1),
                                                          magnesium.LEVEL.values.reshape(-1, 1),
                                                          train_size=0.8, random_state=2)
    lr = LinearRegression()
    model = lr.fit(k_train, mg_train)
    mg_predict = model.predict(k_test)
    # draw scatter plot and linear prediction line
    plt.scatter(k_train, mg_train, alpha=0.4)
    plt.plot(k_test, mg_predict, color='orange', alpha=0.5)
    plt.xlabel('Magnesium (Mg) Milligram/ 100 gram')
    plt.ylabel('Potassium (K) Milligram/ 100 gram')
    plt.legend(['Prediction', 'Datapoints (Train Split)'])
    plt.title('Potassium/ Magnesium Regression Model (Correlation : ~76.%)')
    plt.xlim(left=0, right=2000)
    plt.ylim(0, 280)
    plt.savefig('./plots/model.png')
    # zoomed version
    plt.xlim(left=0, right=700)
    plt.ylim(0, 100)
    plt.savefig('./plots/model_zoom.png')
    plt.cla()

    print(mg_predict)
    print(model.score(k_test, mg_test))


# function calls
nutri_dist()
predict()

# exporting correlations and sorted magnesium and potassium tables as .csv
nutri_pivot.corr().to_csv('./tables/correlations.csv')
nutri_pivot.head(20).to_csv('./tables/nutri_means.csv')
potassium_high.to_csv('./tables/high_potassium.csv')
potassium_low.to_csv('./tables/low_potassium.csv')
magnesium_high.to_csv('./tables/magnesium_high.csv')
magnesium_low.to_csv('./tables/magnesium_low.csv')

# exporting correlations and sorted magnesium and potassium tables as .xlsx
nutri_pivot.corr().to_excel('./tables/excel/correlations.xlsx')
nutri_pivot.head(20).to_excel('./tables/excel/nutri_means.xlsx')
potassium_high.to_excel('./tables/excel/high_potassium.xlsx')
potassium_low.to_excel('./tables/excel/low_potassium.xlsx')
magnesium_high.to_excel('./tables/excel/magnesium_high.xlsx')
magnesium_low.to_excel('./tables/excel/magnesium_low.xlsx')

'''print(len(grp_rec_lv1))
print(grp_rec_lv1.head(100).to_string())'''

# print(df.dtypes, df.efsaprodcode2_recoded.unique())
#
# print(nutri_pivot.to_string())
# print(nutri_pivot.corr().to_string())
'''print(potassium_high)
print(magnesium_high)
print(magnesium_low)
print(potassium_low)'''
