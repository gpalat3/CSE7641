import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
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

def runCurves(C, class_weight, max_iter, kernel, X_train, y_train, dataset):
    print('Calculating Learning Curve Accuracy for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc, fit_times_lc, _ = learning_curve(
        SVC(C=C, class_weight=class_weight, max_iter=max_iter, random_state=99, kernel=kernel), X_train, y_train.values.ravel(), cv=5,
        scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40), shuffle=True,
        return_times=True)
    train_scores_lc_mean = np.mean(train_scores_lc, axis=1)
    test_scores_lc_mean = np.mean(test_scores_lc, axis=1)
    fit_times_lc_mean = np.mean(fit_times_lc, axis=1)
    savefile = 'plots/SVM_Learning_Curve_Accuracy_' + kernel + '_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Learning Curve',
                                             'Training Size', 'Accuracy']
    print('Started plotting Learning Curve Accuracy for SVM for ' + dataset)
    plotLearningCurve(train_sizes, train_scores_lc_mean, test_scores_lc_mean,
                        label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Learning Curve Accuracy for SVM for ' + dataset)
    print("--------------------------------")
    print('Calculating Learning Curve MSE for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc = learning_curve(
        SVC(C=C, class_weight=class_weight, max_iter=max_iter, random_state=99, kernel=kernel), X_train, y_train.values.ravel(), cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40), shuffle=True)
    train_scores_lc_mean = -np.mean(train_scores_lc, axis=1)
    test_scores_lc_mean = -np.mean(test_scores_lc, axis=1)
    savefile = 'plots/SVM_Learning_Curve_MSE_' + kernel + '_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Error', 'Cross Validation Error', 'Learning Curve',
                                             'Training Size', 'MSE']
    print('Started plotting Learning Curve MSE SVM for ' + dataset)
    plotLearningCurve(train_sizes, train_scores_lc_mean, test_scores_lc_mean,
                      label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Learning Curve MSE SVM for ' + dataset)
    print("--------------------------------")
    print('Calculating cross val score for incremental max_iter for ' + dataset)
    scores_cv = []
    scores_mean_cv = []
    train_acc_cv = []
    iter = max_iter
    for i in range(1, iter + 1):
        clf = SVC(C=C, class_weight=class_weight, max_iter=i, random_state=99, kernel=kernel)
        scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=5, scoring='accuracy')
        clf.fit(X_train, y_train.values.ravel())
        acc = clf.score(X_train, y_train)
        scores_cv.append(scores)
        scores_mean_cv.append(scores.mean())
        train_acc_cv.append(acc)
    scores_mean_cv = np.array(scores_mean_cv)
    train_acc_cv = np.array(train_acc_cv)
    savefile = 'plots/SVM_Validation_Curve_' + kernel + '_' + str(iter) + '_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Validation Curve',
                                             'Max Iteration', 'Accuracy']
    print('Started plotting Validation Curve for SVM for ' + dataset)
    plotValidationCurve(range(1, iter + 1), scores_mean_cv, train_acc_cv, label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Validation Curve for SVM for ' + dataset)
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

def plotValidationCurve(max_iter, scores_mean_cv, train_acc_cv, label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(max_iter, train_acc_cv, 'o-', label=label1, color='blue')
    plt.plot(max_iter, scores_mean_cv, 'o-', label=label2, color='green')
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
    kernel = 'rbf'
    max_iter = 100
    clf = SVC(random_state=99, kernel=kernel, max_iter=max_iter)
    print('Getting scores for dataset 1 ')
    y_pred_1_svm_rbf, y_pred_train_1_svm_rbf, learn_tm_1_svm_rbf, query_tm_1_svm_rbf, query_tm_train_1_svm_rbf, rmse_1_svm_rbf, rmse_train_1_svm_rbf = mlModel(
        clf, X_train_1, X_test_1, y_train_1, y_test_1)
    print('Getting scores for dataset 2 ')

    y_pred_2_svm_rbf, y_pred_train_2_svm_rbf, learn_tm_2_svm_rbf, query_tm_2_svm_rbf, query_tm_train_2_svm_rbf, rmse_2_svm_rbf, rmse_train_2_svm_rbf = mlModel(
        clf, X_train_2, X_test_2, y_train_2, y_test_2)


    #grid_param = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 5, 10, 20, 50, 100], 'class_weight': ['dict', 'balanced'], 'kernel': ['linear', 'poly', 'rbf']}
    grid_param = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3],
                  'class_weight': [None, 'balanced'], 'max_iter': [1, 10, 100]}
    grid_best_params_1, grid_best_estimator_1, grid_best_score_1 = gridSearch(grid_param, clf, X_train_1, y_train_1,
                                                                              scoring='accuracy', dataset='car')
    grid_best_params_2, grid_best_estimator_2, grid_best_score_2 = gridSearch(grid_param, clf, X_train_2, y_train_2,
                                                                              scoring='accuracy', dataset='adult')
    print(grid_best_params_1, grid_best_estimator_1, grid_best_score_1)
    print(grid_best_params_2, grid_best_estimator_2, grid_best_score_2)
    C = grid_best_params_1['C']
    class_weight = grid_best_params_1['class_weight']
    max_iter = grid_best_params_1['max_iter']
    clf = SVC(random_state=99, C=C, class_weight=class_weight, max_iter=max_iter, kernel=kernel)
    y_pred_1_hyper_svm_rbf, y_pred_train_1_hyper_svm_rbf, learn_tm_1_hyper_svm_rbf, query_tm_1_hyper_svm_rbf, query_tm_train_1_hyper_svm_rbf, rmse_1_hyper_svm_rbf, rmse_train_1_hyper_svm_rbf = mlModel(
        clf,
        X_train_1,
        X_test_1,
        y_train_1,
        y_test_1)
    runCurves(C, class_weight, max_iter, kernel, X_train_1, y_train_1, dataset='car')
    print("--------------------------------")
    print("--------------------------------")
    print("SVM RBF Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_svm_rbf))
    print("SVM RBF Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_svm_rbf))
    print("SVM RBF Learning Time Dataset 1:", learn_tm_1_svm_rbf)
    print("SVM RBF Query Time Dataset 1:", query_tm_1_svm_rbf)
    print("SVM RBF Hypertuning Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_hyper_svm_rbf))
    print("SVM RBF Hypertuning Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_hyper_svm_rbf))
    print("SVM RBF Hypertuning RMSE Dataset 1: Out Sample", rmse_1_hyper_svm_rbf)
    print("SVM RBF Hypertuning RMSE Dataset 1: In Sample", rmse_train_1_hyper_svm_rbf)
    print("SVM RBF Hypertuning Learning Time Dataset 1:", learn_tm_1_hyper_svm_rbf)
    print("SVM RBF Hypertuning Query Time Dataset 1:", query_tm_1_hyper_svm_rbf)
    print("--------------------------------")
    print("--------------------------------")
    C = grid_best_params_2['C']
    class_weight = grid_best_params_2['class_weight']
    max_iter = grid_best_params_2['max_iter']
    clf = SVC(random_state=99, C=C, class_weight=class_weight, max_iter=max_iter, kernel=kernel)
    y_pred_2_hyper_svm_rbf, y_pred_train_2_hyper_svm_rbf, learn_tm_2_hyper_svm_rbf, query_tm_2_hyper_svm_rbf, query_tm_train_2_hyper_svm_rbf, rmse_2_hyper_svm_rbf, rmse_train_2_hyper_svm_rbf = mlModel(
        clf,
        X_train_2,
        X_test_2,
        y_train_2,
        y_test_2)
    runCurves(C, class_weight, max_iter, kernel, X_train_2, y_train_2, dataset='adult')
    print("SVM RBF Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_svm_rbf))
    print("SVM RBF Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_svm_rbf))
    print("SVM RBF Learning Time Dataset 2:", learn_tm_2_svm_rbf)
    print("SVM RBF Query Time Dataset 2:", query_tm_2_svm_rbf)
    print("SVM RBF Hypertuning Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_hyper_svm_rbf))
    print("SVM RBF Hypertuning Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_hyper_svm_rbf))
    print("SVM RBF Hypertuning RMSE Dataset 2: Out Sample", rmse_2_hyper_svm_rbf)
    print("SVM RBF Hypertuning RMSE Dataset 2: In Sample", rmse_train_2_hyper_svm_rbf)
    print("SVM RBF Hypertuning Learning Time Dataset 2:", learn_tm_2_hyper_svm_rbf)
    print("SVM RBF Hypertuning Query Time Dataset 2:", query_tm_2_hyper_svm_rbf)
    print("--------------------------------")
    print("--------------------------------")
    print(grid_best_params_1, grid_best_estimator_1, grid_best_score_1)
    print(grid_best_params_2, grid_best_estimator_2, grid_best_score_2)

'''
Result: 

Grid Search for dataset car
{'C': 0.75, 'class_weight': None, 'max_iter': 100} SVC(C=0.75, max_iter=100, random_state=99) 0.7286924316724392

Grid Search for dataset adult
{'C': 0.25, 'class_weight': None, 'max_iter': 10} SVC(C=0.25, max_iter=10, random_state=99) 0.5219196684183897
--------------------------------
SVM RBF Accuracy Dataset 1: Out Sample 0.7167630057803468
SVM RBF Accuracy Dataset 1: In Sample 0.7568238213399504
SVM RBF Learning Time Dataset 1: 0.031280517578125
SVM RBF Query Time Dataset 1: 0.015629291534423828
SVM RBF Hypertuning Accuracy Dataset 1: Out Sample 0.7109826589595376
SVM RBF Hypertuning Accuracy Dataset 1: In Sample 0.7543424317617866
SVM RBF Hypertuning RMSE Dataset 1: Out Sample 0.7925471728513105
SVM RBF Hypertuning RMSE Dataset 1: In Sample 0.7405316311773545
SVM RBF Hypertuning Learning Time Dataset 1: 0.03023672103881836
SVM RBF Hypertuning Query Time Dataset 1: 0.01015925407409668
--------------------------------
SVM RBF Accuracy Dataset 2: Out Sample 0.19981574367898455
SVM RBF Accuracy Dataset 2: In Sample 0.20024570024570024
SVM RBF Learning Time Dataset 2: 0.9330954551696777
SVM RBF Query Time Dataset 2: 0.2848351001739502
SVM RBF Hypertuning Accuracy Dataset 2: Out Sample 0.3407718292558092
SVM RBF Hypertuning Accuracy Dataset 2: In Sample 0.342005967005967
SVM RBF Hypertuning RMSE Dataset 2: Out Sample 0.8119286734339358
SVM RBF Hypertuning RMSE Dataset 2: In Sample 0.811168313603307
SVM RBF Hypertuning Learning Time Dataset 2: 0.15285134315490723
SVM RBF Hypertuning Query Time Dataset 2: 0.04055380821228027
--------------------------------
--------------------------------
'''