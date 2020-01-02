from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

dataset = load_breast_cancer()
feature_train, feature_validation, target_train, target_validation = train_test_split(dataset.data, dataset.target)
best_gamma_C = []
best_score = 0
for i in range(1, 101):
    for j in range(1, 101):
        model = SVC(gamma=i, C=j)
        model.fit(feature_train, target_train)
        target_predict = model.predict(feature_validation)
        score = model.score(feature_validation, target_validation)
        print(i, j, score)
        if score > best_score:
            best_score = score
            best_gamma_C = [i, j]

print('Best gamma: ', best_gamma_C[0], 'Best C: ', best_gamma_C[1], 'Score: ', best_score)
