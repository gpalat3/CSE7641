import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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
    print('Started grid Search for dataset ' + dataset)
    grid_search.fit(X, y.values.ravel())
    print('Ended grid Search for dataset ' + dataset)
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

def runKnnCurves(k, X_train, y_train, dataset):
    print('Calculating Learning Curve Accuracy for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc, fit_times_lc, _ = learning_curve(
        KNeighborsClassifier(), X_train, y_train.values.ravel(), cv=5,
        scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40), shuffle=True,
        return_times=True)
    train_scores_lc_mean = np.mean(train_scores_lc, axis=1)
    test_scores_lc_mean = np.mean(test_scores_lc, axis=1)
    fit_times_lc_mean = np.mean(fit_times_lc, axis=1)
    savefile = 'plots/KNN_Learning_Curve_Accuracy_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Learning Curve',
                                             'Training Size', 'Accuracy']
    print('Started plotting Learning Curve Accuracy for KNN for ' + dataset)
    plotLearningCurve(train_sizes, train_scores_lc_mean, test_scores_lc_mean,
                        label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Learning Curve Accuracy for KNN for ' + dataset)
    print("--------------------------------")
    print('Calculating Learning Curve MSE for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc = learning_curve(
        KNeighborsClassifier(), X_train, y_train.values.ravel(), cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40), shuffle=True)
    train_scores_lc_mean = -np.mean(train_scores_lc, axis=1)
    test_scores_lc_mean = -np.mean(test_scores_lc, axis=1)
    savefile = 'plots/KNN_Learning_Curve_MSE_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Error', 'Cross Validation Error', 'Learning Curve',
                                             'Training Size', 'MSE']
    print('Started plotting Learning Curve MSE KNN for ' + dataset)
    plotLearningCurve(train_sizes, train_scores_lc_mean, test_scores_lc_mean,
                      label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Learning Curve MSE KNN for ' + dataset)
    print("--------------------------------")
    print('Calculating cross val score for incremental k for ' + dataset)
    scores_cv = []
    scores_mean_cv = []
    train_acc_cv = []
    for i in k:
        clf = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=5, scoring='accuracy')
        clf.fit(X_train, y_train.values.ravel())
        acc = clf.score(X_train, y_train)
        scores_cv.append(scores)
        scores_mean_cv.append(scores.mean())
        train_acc_cv.append(acc)
    scores_mean_cv = np.array(scores_mean_cv)
    train_acc_cv = np.array(train_acc_cv)
    savefile = 'plots/KNN_Validation_Curve_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Validation Curve',
                                             'K', 'Accuracy']
    print('Started plotting Validation Curve for KNN for ' + dataset)
    plotValidationCurve(k, scores_mean_cv, train_acc_cv, label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Validation Curve for KNN for ' + dataset)
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

def plotValidationCurve(k, scores_mean_cv, train_acc_cv, label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(k, train_acc_cv, 'o-', label=label1, color='blue')
    plt.plot(k, scores_mean_cv, 'o-', label=label2, color='green')
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
    clf = KNeighborsClassifier()
    print('Getting scores for dataset 1 ')
    y_pred_1_knn, y_pred_train_1_knn, learn_tm_1_knn, query_tm_1_knn, query_tm_train_1_knn, rmse_1_knn, rmse_train_1_knn = mlModel(
        clf, X_train_1, X_test_1, y_train_1, y_test_1)
    print('Getting scores for dataset 2 ')
    y_pred_2_knn, y_pred_train_2_knn, learn_tm_2_knn, query_tm_2_knn, query_tm_train_2_knn, rmse_2_knn, rmse_train_2_knn = mlModel(
        clf, X_train_2, X_test_2, y_train_2, y_test_2)
    k = range(1, 36)
    runKnnCurves(k, X_train_1, y_train_1, dataset='car')
    runKnnCurves(k, X_train_2, y_train_2, dataset='adult')
    grid_param = {'n_neighbors': k}
    grid_best_params_1, grid_best_estimator_1, grid_best_score_1 = gridSearch(grid_param, clf, X_train_1,
                                                                              y_train_1, scoring='accuracy',
                                                                              dataset='car')
    print('Grid search results for dataset 1: ', grid_best_params_1, grid_best_estimator_1, grid_best_score_1)
    
    grid_best_params_2, grid_best_estimator_2, grid_best_score_2 = gridSearch(grid_param, clf, X_train_2,
                                                                              y_train_2, scoring='accuracy',
                                                                              dataset='adult')
    print('Grid search results for dataset 2: ', grid_best_params_2, grid_best_estimator_2, grid_best_score_2)
    clf = KNeighborsClassifier(n_neighbors=grid_best_params_1['n_neighbors'])
    print('Getting scores for dataset 1 with best params ')
    y_pred_1_hyper_knn, y_pred_train_1_hyper_knn, learn_tm_1_hyper_knn, query_tm_1_hyper_knn, query_tm_train_1_hyper_knn, rmse_1_hyper_knn, rmse_train_1_hyper_knn = mlModel(
        clf, X_train_1, X_test_1, y_train_1, y_test_1)
    clf = KNeighborsClassifier(n_neighbors=grid_best_params_2['n_neighbors'])
    print('Getting scores for dataset 2 with best params ')
    y_pred_2_hyper_knn, y_pred_train_2_hyper_knn, learn_tm_2_hyper_knn, query_tm_2_hyper_knn, query_tm_train_2_hyper_knn, rmse_2_hyper_knn, rmse_train_2_hyper_knn = mlModel(
        clf, X_train_2, X_test_2, y_train_2, y_test_2)
    print("--------------------------------")
    print("--------------------------------")
    print("KNN Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_knn))
    print("KNN Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_knn))
    print("KNN Hypertuning Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_hyper_knn))
    print("KNN Hypertuning Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_hyper_knn))
    print("--------------------------------")
    print("KNN Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_knn))
    print("KNN Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_knn))
    print("KNN Hypertuning Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_hyper_knn))
    print("KNN Hypertuning Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_hyper_knn))
    print("--------------------------------")
    print("--------------------------------")

'''
Result: 

Grid search results for dataset 1:  {'n_neighbors': 1} KNeighborsClassifier(n_neighbors=1) 0.7427523061623401
Started grid Search for dataset adult
Ended grid Search for dataset adult
Grid search results for dataset 2:  {'n_neighbors': 16} KNeighborsClassifier(n_neighbors=16) 0.8403828368556917
Getting scores for dataset 1 with best params 
Getting scores for dataset 2 with best params 
--------------------------------
--------------------------------
KNN Accuracy Dataset 1: Out Sample 0.7572254335260116
KNN Accuracy Dataset 1: In Sample 0.7899090157154673
KNN Hypertuning Accuracy Dataset 1: Out Sample 0.7552986512524085
KNN Hypertuning Accuracy Dataset 1: In Sample 0.7568238213399504
--------------------------------
KNN Accuracy Dataset 2: Out Sample 0.8243423072985976
KNN Accuracy Dataset 2: In Sample 0.8563969813969814
KNN Hypertuning Accuracy Dataset 2: Out Sample 0.8410277408127751
KNN Hypertuning Accuracy Dataset 2: In Sample 0.8515268515268515
--------------------------------
--------------------------------
'''