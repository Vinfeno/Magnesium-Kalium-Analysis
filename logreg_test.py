from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

breast_cancer = load_breast_cancer()
print(breast_cancer.keys())


def knn_predict(k):
    train_data, validation_data, train_target, validation_target = train_test_split(breast_cancer.data,
                                                                                    breast_cancer.target
                                                                                    , train_size=0.8, test_size=0.2)
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn_model.fit(train_data, train_target)
    print(knn_model.score(train_data, train_target))
    print(knn_model.score(validation_data, validation_target))

    pred = knn_model.predict(validation_data)
    print(pred)
    correct = 0
    false = 0
    for i in range(len(pred)):
        if pred[i] == validation_target[i]:
            correct += 1
        else:
            false += 1
    print(correct, false, (correct / len(pred)))


def knn_accuracy():
    accuracies = []
    for i in range(1, 101):
        train_data, validation_data, train_target, validation_target = train_test_split(breast_cancer.data,
                                                                                        breast_cancer.target
                                                                                        , train_size=0.8, test_size=0.2)
        knn_model = KNeighborsClassifier(n_neighbors=i, weights='uniform')
        knn_model.fit(train_data, train_target)
        accuracies.append([i, knn_model.score(validation_data, validation_target)])
    return accuracies


def mean_accuracy():
    best_outer = []
    for x in range(100):
        print('iteration #: ', x)
        acc = knn_accuracy()
        best_inner = [0, 0]
        for y in acc:
            if y[1] > best_inner[1]:
                best_inner = y
                print('new best: ', y)
        best_outer.append(best_inner[0])
        print('best in set: ', best_inner)
    total = 0
    for z in best_outer:
        total += z
    mean = total / len(best_outer)
    print('mean k: ', mean)
    return int(round(mean, 0))


def logreg_predict():
    scale = StandardScaler()
    data_normalized = scale.fit_transform(breast_cancer.data)
    train_data, validation_data, train_target, validation_target = train_test_split(data_normalized,
                                                                                    breast_cancer.target
                                                                                    , train_size=0.8, test_size=0.2)
    model = LogisticRegression()
    model.fit(train_data, train_target)
    prediction = model.predict(validation_data)
    probabilities = model.predict_proba(validation_data)
    print(model.score(train_data, train_target))
    print(model.score(validation_data, validation_target))
    coeffs = pd.DataFrame({'features': breast_cancer.feature_names, 'coefficients': model.coef_[0]})
    coeffs['abs'] = coeffs.coefficients.apply(lambda x: abs(x))
    return coeffs


print(logreg_predict().sort_values('abs', ascending=0))
print(breast_cancer.DESCR)
