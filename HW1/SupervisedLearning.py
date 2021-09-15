import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV
from sklearn import metrics
pd.options.mode.chained_assignment = None

def loadAdultData(filename, col_names):
    df = pd.read_csv(filename, encoding='utf-8', header=None, names=col_names)
    df = df.dropna()
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    label = ['label']
    X = pd.get_dummies(df[features])
    y = df[label]
    y.replace(' <=50K', 0, inplace=True)
    y.replace(' >50K', 1, inplace=True)
    return X, y

def loadCarData(filename, col_names):
    df = pd.read_csv(filename, encoding='utf-8', header=None, names=col_names)
    df = df.dropna()
    features = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety']
    label = ['class']
    X = pd.get_dummies(df[features])
    y = df[label]
    y.replace('unacc', 1, inplace=True)
    y.replace('acc', 2, inplace=True)
    y.replace('good', 3, inplace=True)
    y.replace('vgood', 4, inplace=True)
    return X, y

def gridSearch(grid_param, clf, X, y, scoring, dataset):
    grid_search = GridSearchCV(estimator=clf, param_grid=grid_param, scoring=scoring, cv=5, n_jobs=-1,
                               return_train_score=True)
    print('Grid Search for dataset ' + dataset)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_estimator_, grid_search.best_score_

def decisionTree(clf, X_train, X_test, y_train, y_test):
    dt_learn_start_tm = time.time()
    clf = clf.fit(X_train, y_train)
    dt_learn_end_tm = time.time()
    dt_learn_tm = dt_learn_end_tm - dt_learn_start_tm
    dt_query_start_tm = time.time()
    y_pred = clf.predict(X_test)
    dt_query_end_tm = time.time()
    dt_query_tm = dt_query_end_tm - dt_query_start_tm
    dt_query_start_tm_train = time.time()
    y_pred_train = clf.predict(X_train)
    dt_query_end_tm_train = time.time()
    dt_query_tm_train = dt_query_end_tm_train - dt_query_start_tm_train
    rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    rmse_train = math.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
    return y_pred, y_pred_train, dt_learn_tm, dt_query_tm, dt_query_tm_train, rmse, rmse_train

def runCurves(depth, X_train, y_train, dataset):
    print('Calculating Learning Curve Accuracy for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc, fit_times_lc, _ = learning_curve(
        DecisionTreeClassifier(random_state=99), X_train, y_train, cv=5,
        scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 30), shuffle=True,
        return_times=True)
    train_scores_lc_mean = np.mean(train_scores_lc, axis=1)
    test_scores_lc_mean = np.mean(test_scores_lc, axis=1)
    fit_times_lc_mean = np.mean(fit_times_lc, axis=1)
    savefile = 'plots/DT_Learning_Curve_Accuracy_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Learning Curve',
                                             'Training Size', 'Accuracy']
    print('Started plotting Learning Curve Accuracy for decision tree for ' + dataset)
    plotLearningCurve(train_sizes, train_scores_lc_mean, test_scores_lc_mean,
                        label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Learning Curve Accuracy for decision tree for ' + dataset)
    print("--------------------------------")
    print('Calculating Learning Curve MSE for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc = learning_curve(
        DecisionTreeClassifier(random_state=99), X_train, y_train, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 30), shuffle=True)
    train_scores_lc_mean = -np.mean(train_scores_lc, axis=1)
    test_scores_lc_mean = -np.mean(test_scores_lc, axis=1)
    savefile = 'plots/DT_Learning_Curve_MSE_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Error', 'Cross Validation Error', 'Learning Curve',
                                             'Training Size', 'MSE']
    print('Started plotting Learning Curve MSE decision tree for ' + dataset)
    plotLearningCurve(train_sizes, train_scores_lc_mean, test_scores_lc_mean,
                      label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Learning Curve MSE decision tree for ' + dataset)
    print("--------------------------------")
    print('Calculating cross val score for incremental tree depth for ' + dataset)
    scores_cv = []
    scores_mean_cv = []
    train_acc_cv = []
    for i in depth:
        clf_DT = DecisionTreeClassifier(max_depth=i)
        scores = cross_val_score(clf_DT, X_train, y_train, cv=5, scoring='accuracy')
        DT = clf_DT.fit(X_train, y_train)
        acc = DT.score(X_train, y_train)
        scores_cv.append(scores)
        scores_mean_cv.append(scores.mean())
        train_acc_cv.append(acc)
    scores_mean_cv = np.array(scores_mean_cv)
    train_acc_cv = np.array(train_acc_cv)
    savefile = 'plots/DT_Validation_Curve_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Validation Curve',
                                             'Depth Of Tree', 'Accuracy']
    print('Started plotting Validation Curve for decision tree for ' + dataset)
    plotValidationCurve(depth, scores_mean_cv, train_acc_cv, label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Validation Curve for decision tree for ' + dataset)

def plotLearningCurve(train_sizes, train_scores_mean, test_scores_mean,
                        label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(train_sizes, train_scores_mean, 'o-', label=label1, color='blue')
    plt.plot(train_sizes, test_scores_mean, 'o-', label=label2, color='green')
    plt.legend(loc='best')
    plt.savefig(savefile)

def plotFitTimes():
    return None

def plotValidationCurve(depth, scores_mean_cv, train_acc_cv, label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(depth, train_acc_cv, 'o-', label=label1, color='blue')
    plt.plot(depth, scores_mean_cv, 'o-', label=label2, color='green')
    plt.legend(loc='best')
    plt.savefig(savefile)
    return None

if __name__ == '__main__':
    col_names_car = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    X_car, y_car = loadCarData('data/car_evaluation/car_data.csv', col_names_car)
    col_names_adult = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label']
    X_adult, y_adult = loadAdultData('data/adult/adult_data.csv', col_names_adult)
    X_train_car, X_test_car, y_train_car, y_test_car = train_test_split(X_car, y_car, test_size=0.3,
                                                                        random_state=99)
    X_train_adult, X_test_adult, y_train_adult, y_test_adult = train_test_split(X_adult, y_adult, test_size=0.3,
                                                                        random_state=99)
    clf_DT = DecisionTreeClassifier(random_state=99)
    y_pred, y_pred_train, dt_learn_tm, dt_query_tm, dt_query_tm_train, rmse, rmse_train = decisionTree(clf_DT,
                                                                                                       X_train_car,
                                                                                                       X_test_car,
                                                                                                       y_train_car,
                                                                                                       y_test_car)
    print("--------------------------------")
    print("--------------------------------")
    print("Accuracy: Out Sample", metrics.accuracy_score(y_test_car, y_pred))
    print("Accuracy: In Sample", metrics.accuracy_score(y_train_car, y_pred_train))
    print("RMSE: Out Sample", rmse)
    print("RMSE: In Sample", rmse_train)
    print("Learning Time:", dt_learn_tm)
    print("Query Time:", dt_query_tm)
    print("--------------------------------")
    print("--------------------------------")
    y_pred, y_pred_train, dt_learn_tm, dt_query_tm, dt_query_tm_train, rmse, rmse_train = decisionTree(clf_DT,
                                                                                                       X_train_adult,
                                                                                                       X_test_adult,
                                                                                                       y_train_adult,
                                                                                                       y_test_adult)
    print("Accuracy: Out Sample", metrics.accuracy_score(y_test_adult, y_pred))
    print("Accuracy: In Sample", metrics.accuracy_score(y_train_adult, y_pred_train))
    print("RMSE: Out Sample", rmse)
    print("RMSE: In Sample", rmse_train)
    print("Learning Time:", dt_learn_tm)
    print("Query Time:", dt_query_tm)
    print("--------------------------------")
    print("--------------------------------")
    depth = range(1, 36)
    runCurves(depth, X_train_car, y_train_car, dataset='car')
    runCurves(depth, X_train_adult, y_train_adult, dataset='adult')
    grid_param = {'max_depth': depth, 'criterion': ['gini', 'entropy']}
    grid_best_params_car, grid_best_estimator_car, grid_best_score_car = gridSearch(grid_param, clf_DT, X_train_car, y_train_car, scoring='accuracy', dataset='car')
    print(grid_best_params_car, grid_best_estimator_car, grid_best_score_car)
    grid_best_params_adult, grid_best_estimator_adult, grid_best_score_adult = gridSearch(grid_param, clf_DT, X_train_adult, y_train_adult, scoring='accuracy', dataset='adult')
    print(grid_best_params_adult, grid_best_estimator_adult, grid_best_score_adult)
    best_params_car = {'max_depth': grid_best_params_car['max_depth'], 'criterion': grid_best_params_car['criterion']}
    best_params_adult = {'max_depth': grid_best_params_adult['max_depth'], 'criterion': grid_best_params_adult['criterion']}
    print(best_params_car)
    print(best_params_adult)


