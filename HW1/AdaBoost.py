import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV
from sklearn import metrics
pd.options.mode.chained_assignment = None

def loadAdultData(filename, col_names):
    df = pd.read_csv(filename, encoding='utf-8', header=None, names=col_names)
    df = df.dropna()
    #features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
    #            'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    features = ['workclass', 'education', 'education_num', 'occupation',
                'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    label = ['label']
    X = pd.get_dummies(df[features])
    y = df[label]
    y.replace(' <=50K', 0, inplace=True)
    y.replace(' >50K', 1, inplace=True)
    return X, y

def loadCarData(filename, col_names):
    df = pd.read_csv(filename, encoding='utf-8', header=None, names=col_names)
    df = df.dropna()
    #features = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety']
    features = ['buying_cost', 'maintenance_cost', 'lug_boot', 'safety']
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
    grid_search.fit(X, y.values.ravel())
    return grid_search.best_params_, grid_search.best_estimator_, grid_search.best_score_

def mlModel(clf, X_train, X_test, y_train, y_test):
    dt_learn_start_tm = time.time()
    clf.fit(X_train, y_train.values.ravel())
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

def runDTCurves(estimators, n_estimators, learning_rate, X_train, y_train, dataset):
    print('Calculating Learning Curve Accuracy for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc, fit_times_lc, _ = learning_curve(
        AdaBoostClassifier(random_state=99, n_estimators=n_estimators, learning_rate=learning_rate), X_train, y_train.values.ravel(), cv=5,
        scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40), shuffle=True,
        return_times=True)
    train_scores_lc_mean = np.mean(train_scores_lc, axis=1)
    test_scores_lc_mean = np.mean(test_scores_lc, axis=1)
    fit_times_lc_mean = np.mean(fit_times_lc, axis=1)
    savefile = 'plots/Ada_Learning_Curve_Accuracy_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Learning Curve',
                                             'Training Size', 'Accuracy']
    print('Started plotting Learning Curve Accuracy for Ada for' + dataset)
    plotLearningCurve(train_sizes, train_scores_lc_mean, test_scores_lc_mean,
                        label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Learning Curve Accuracy for Ada for ' + dataset)
    print("--------------------------------")
    print('Calculating Learning Curve MSE for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc = learning_curve(
        AdaBoostClassifier(random_state=99, n_estimators=n_estimators, learning_rate=learning_rate), X_train, y_train.values.ravel(), cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40), shuffle=True)
    train_scores_lc_mean = -np.mean(train_scores_lc, axis=1)
    test_scores_lc_mean = -np.mean(test_scores_lc, axis=1)
    savefile = 'plots/Ada_Learning_Curve_MSE_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Error', 'Cross Validation Error', 'Learning Curve',
                                             'Training Size', 'MSE']
    print('Started plotting Learning Curve MSE Ada for ' + dataset)
    plotLearningCurve(train_sizes, train_scores_lc_mean, test_scores_lc_mean,
                      label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Learning Curve MSE Ada for ' + dataset)
    print("--------------------------------")
    print('Calculating cross val score for incremental number of estimators for ' + dataset)
    scores_cv = []
    scores_mean_cv = []
    train_acc_cv = []
    for i in estimators:
        clf = AdaBoostClassifier(random_state=99, n_estimators=i, learning_rate=learning_rate)
        scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=5, scoring='accuracy')
        clf.fit(X_train, y_train.values.ravel())
        acc = clf.score(X_train, y_train)
        scores_cv.append(scores)
        scores_mean_cv.append(scores.mean())
        train_acc_cv.append(acc)
    scores_mean_cv = np.array(scores_mean_cv)
    train_acc_cv = np.array(train_acc_cv)
    savefile = 'plots/Ada_Validation_Curve_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Validation Curve',
                                             'Estimators', 'Accuracy']
    print('Started plotting Validation Curve for ' + dataset)
    plotValidationCurve(estimators, scores_mean_cv, train_acc_cv, label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Validation Curve for ' + dataset)
    print("--------------------------------")

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

def plotValidationCurve(depth, scores_mean_cv, train_acc_cv, label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(depth, train_acc_cv, 'o-', label=label1, color='blue')
    plt.plot(depth, scores_mean_cv, 'o-', label=label2, color='green')
    plt.legend(loc='best')
    plt.savefig(savefile)

if __name__ == '__main__':
    col_names_car = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    X_1, y_1 = loadCarData('data/car_evaluation/car_data.csv', col_names_car)
    col_names_adult = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label']
    X_2, y_2 = loadAdultData('data/adult/adult_data.csv', col_names_adult)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.3,
                                                                        random_state=99)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.3,
                                                                        random_state=99)
    clf = AdaBoostClassifier(random_state=99)
    y_pred_1_dt, y_pred_train_1_dt, learn_tm_1_dt, query_tm_1_dt, query_tm_train_1_dt, rmse_1_dt, rmse_train_1_dt = mlModel(clf,
                                                                                                       X_train_1,
                                                                                                       X_test_1,
                                                                                                       y_train_1,
                                                                                                       y_test_1)
    y_pred_2_dt, y_pred_train_2_dt, learn_tm_2_dt, query_tm_2_dt, query_tm_train_2_dt, rmse_2_dt, rmse_train_2_dt = mlModel(clf,
                                                                                                       X_train_2,
                                                                                                       X_test_2,
                                                                                                       y_train_2,
                                                                                                       y_test_2)
    estimators = range(1, 200)
    grid_param = {'n_estimators': estimators, 'learning_rate': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3]}
    grid_best_params_1, grid_best_estimator_1, grid_best_score_1 = gridSearch(grid_param, clf, X_train_1, y_train_1, scoring='accuracy', dataset='car')
    print(grid_best_params_1, grid_best_estimator_1, grid_best_score_1)
    grid_best_params_2, grid_best_estimator_2, grid_best_score_2 = gridSearch(grid_param, clf, X_train_2, y_train_2, scoring='accuracy', dataset='adult')
    print(grid_best_params_2, grid_best_estimator_2, grid_best_score_2)
    n_estimators = grid_best_params_1['n_estimators']
    learning_rate = grid_best_params_1['learning_rate']
    clf = AdaBoostClassifier(random_state=99, n_estimators=n_estimators, learning_rate=learning_rate)
    y_pred_1_hyper_dt, y_pred_train_1_hyper_dt, learn_tm_1_hyper_dt, query_tm_1_hyper_dt, query_tm_train_1_hyper_dt, rmse_1_hyper_dt, rmse_train_1_hyper_dt = mlModel(
        clf,
        X_train_1,
        X_test_1,
        y_train_1,
        y_test_1)
    runDTCurves(estimators, n_estimators, learning_rate, X_train_1, y_train_1, dataset='car')
    print("--------------------------------")
    print("--------------------------------")
    print("Ada Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_dt))
    print("Ada Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_dt))
    print("Ada Learning Time Dataset 1:", learn_tm_1_dt)
    print("Ada Query Time Dataset 1:", query_tm_1_dt)
    print("Ada Hypertuning Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_hyper_dt))
    print("Ada Hypertuning Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_hyper_dt))
    print("Ada Hypertuning RMSE Dataset 1: Out Sample", rmse_1_hyper_dt)
    print("Ada Hypertuning RMSE Dataset 1: In Sample", rmse_train_1_hyper_dt)
    print("Ada Hypertuning Learning Time Dataset 1:", learn_tm_1_hyper_dt)
    print("Ada Hypertuning Query Time Dataset 1:", query_tm_1_hyper_dt)
    print("--------------------------------")
    print("--------------------------------")
    n_estimators = grid_best_params_2['n_estimators']
    learning_rate = grid_best_params_2['learning_rate']
    clf = AdaBoostClassifier(random_state=99, n_estimators=n_estimators, learning_rate=learning_rate)
    y_pred_2_hyper_dt, y_pred_train_2_hyper_dt, learn_tm_2_hyper_dt, query_tm_2_hyper_dt, query_tm_train_2_hyper_dt, rmse_2_hyper_dt, rmse_train_2_hyper_dt = mlModel(
        clf,
        X_train_2,
        X_test_2,
        y_train_2,
        y_test_2)
    runDTCurves(estimators, n_estimators, learning_rate, X_train_2, y_train_2, dataset='adult')
    print("Ada Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_dt))
    print("Ada Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_dt))
    print("Ada Learning Time Dataset 2:", learn_tm_2_dt)
    print("Ada Query Time Dataset 2:", query_tm_2_dt)
    print("Ada Hypertuning Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_hyper_dt))
    print("Ada Hypertuning Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_hyper_dt))
    print("Ada Hypertuning RMSE Dataset 2: Out Sample", rmse_2_hyper_dt)
    print("Ada Hypertuning RMSE Dataset 2: In Sample", rmse_train_2_hyper_dt)
    print("Ada Hypertuning Learning Time Dataset 2:", learn_tm_2_hyper_dt)
    print("Ada Hypertuning Query Time Dataset 2:", query_tm_2_hyper_dt)
    print("--------------------------------")
    print("--------------------------------")

'''
Result: 

Grid Search for dataset car
{'learning_rate': 0.25, 'n_estimators': 11} AdaBoostClassifier(learning_rate=0.25, n_estimators=11, random_state=99) 0.7105071842529406
Grid Search for dataset adult
{'learning_rate': 1.5, 'n_estimators': 128} AdaBoostClassifier(learning_rate=1.5, n_estimators=128, random_state=99) 0.8448580220849722
--------------------------------
Ada Accuracy Dataset 1: Out Sample 0.29865125240847784
Ada Accuracy Dataset 1: In Sample 0.27791563275434245
Ada Learning Time Dataset 1: 0.0781240463256836
Ada Query Time Dataset 1: 0.0
Ada Hypertuning Accuracy Dataset 1: Out Sample 0.7013487475915221
Ada Hypertuning Accuracy Dataset 1: In Sample 0.7113316790736146
Ada Hypertuning RMSE Dataset 1: Out Sample 0.7815305372998405
Ada Hypertuning RMSE Dataset 1: In Sample 0.8052762319602711
Ada Hypertuning Learning Time Dataset 1: 0.015636682510375977
Ada Hypertuning Query Time Dataset 1: 0.015613555908203125
--------------------------------
--------------------------------
Ada Accuracy Dataset 2: Out Sample 0.8388780837342614
Ada Accuracy Dataset 2: In Sample 0.8395050895050895
Ada Learning Time Dataset 2: 1.5430006980895996
Ada Query Time Dataset 2: 0.17188191413879395
Ada Hypertuning Accuracy Dataset 2: Out Sample 0.8449175964786569
Ada Hypertuning Accuracy Dataset 2: In Sample 0.8471393471393471
Ada Hypertuning RMSE Dataset 2: Out Sample 0.3938050323717855
Ada Hypertuning RMSE Dataset 2: In Sample 0.3909739797744255
Ada Hypertuning Learning Time Dataset 2: 3.687225103378296
Ada Hypertuning Query Time Dataset 2: 0.42065930366516113
--------------------------------
--------------------------------

'''