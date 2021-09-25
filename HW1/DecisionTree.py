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
    grid_search.fit(X, y)
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

def runDTCurves(depth, max_depth, criterion, X_train, y_train, dataset):
    print('Calculating Learning Curve Accuracy for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc, fit_times_lc, _ = learning_curve(
        DecisionTreeClassifier(random_state=99, max_depth=max_depth, criterion=criterion), X_train, y_train, cv=5,
        scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40), shuffle=True,
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
        DecisionTreeClassifier(random_state=99, max_depth=max_depth, criterion=criterion), X_train, y_train, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40), shuffle=True)
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
    scores_cv_gini = []
    scores_mean_cv_gini = []
    train_acc_cv_gini = []
    for i in depth:
        clf = DecisionTreeClassifier(max_depth=i, criterion='gini')
        scores_gini = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)
        acc = clf.score(X_train, y_train)
        scores_cv_gini.append(scores_gini)
        scores_mean_cv_gini.append(scores_gini.mean())
        train_acc_cv_gini.append(acc)
    scores_mean_cv_gini = np.array(scores_mean_cv_gini)
    train_acc_cv_gini = np.array(train_acc_cv_gini)
    scores_cv_entropy = []
    scores_mean_cv_entropy = []
    train_acc_cv_entropy = []
    for i in depth:
        clf = DecisionTreeClassifier(max_depth=i, criterion='entropy')
        scores_entropy = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)
        acc = clf.score(X_train, y_train)
        scores_cv_entropy.append(scores_entropy)
        scores_mean_cv_entropy.append(scores_entropy.mean())
        train_acc_cv_entropy.append(acc)
    scores_mean_cv_entropy = np.array(scores_mean_cv_entropy)
    train_acc_cv_entropy = np.array(train_acc_cv_entropy)
    savefile = 'plots/DT_Validation_Curve_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Validation Curve ' + dataset,
                                             'Depth Of Tree', 'Accuracy']
    print('Started plotting Validation Curve for decision tree for ' + dataset)
    plotValidationCurve(depth, scores_mean_cv_gini, train_acc_cv_gini, scores_mean_cv_entropy, train_acc_cv_entropy, label1, label2, title, xlabel, ylabel, savefile)
    print('Finished plotting Validation Curve for decision tree for ' + dataset)
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

def plotValidationCurve(depth, scores_mean_cv_gini, train_acc_cv_gini, scores_mean_cv_entropy, train_acc_cv_entropy, label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(depth, train_acc_cv_gini, 'o-', label=label1 + ' gini', color='blue')
    plt.plot(depth, scores_mean_cv_gini, 'o-', label=label2 + ' gini', color='green')
    plt.plot(depth, train_acc_cv_entropy, 'o-', label=label1 + ' entropy', color='purple')
    plt.plot(depth, scores_mean_cv_entropy, 'o-', label=label2 + ' entropy', color='red')
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
    clf = DecisionTreeClassifier(random_state=99)
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
    depth = range(1, 36)
    grid_param = {'max_depth': depth, 'criterion': ['gini', 'entropy']}
    grid_best_params_1, grid_best_estimator_1, grid_best_score_1 = gridSearch(grid_param, clf, X_train_1, y_train_1, scoring='accuracy', dataset='car')
    print(grid_best_params_1, grid_best_estimator_1, grid_best_score_1)
    grid_best_params_2, grid_best_estimator_2, grid_best_score_2 = gridSearch(grid_param, clf, X_train_2, y_train_2, scoring='accuracy', dataset='adult')
    print(grid_best_params_2, grid_best_estimator_2, grid_best_score_2)
    max_depth = grid_best_params_1['max_depth']
    criterion = grid_best_params_1['criterion']
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    runDTCurves(depth, max_depth, criterion, X_train_1, y_train_1, dataset='car')
    y_pred_1_hyper_dt, y_pred_train_1_hyper_dt, learn_tm_1_hyper_dt, query_tm_1_hyper_dt, query_tm_train_1_hyper_dt, rmse_1_hyper_dt, rmse_train_1_hyper_dt = mlModel(
        clf,
        X_train_1,
        X_test_1,
        y_train_1,
        y_test_1)
    print("--------------------------------")
    print("--------------------------------")
    print("DT Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_dt))
    print("DT Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_dt))
    print("DT Learning Time Dataset 1:", learn_tm_1_dt)
    print("DT Query Time Dataset 1:", query_tm_1_dt)
    print("DT Hypertuning Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_hyper_dt))
    print("DT Hypertuning Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_hyper_dt))
    print("DT Hypertuning RMSE Dataset 1: Out Sample", rmse_1_hyper_dt)
    print("DT Hypertuning RMSE Dataset 1: In Sample", rmse_train_1_hyper_dt)
    print("DT Hypertuning Learning Time Dataset 1:", learn_tm_1_hyper_dt)
    print("DT Hypertuning Query Time Dataset 1:", query_tm_1_hyper_dt)
    print("--------------------------------")
    print("--------------------------------")
    max_depth = grid_best_params_2['max_depth']
    criterion = grid_best_params_2['criterion']
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    runDTCurves(depth, max_depth, criterion, X_train_2, y_train_2, dataset='adult')
    y_pred_2_hyper_dt, y_pred_train_2_hyper_dt, learn_tm_2_hyper_dt, query_tm_2_hyper_dt, query_tm_train_2_hyper_dt, rmse_2_hyper_dt, rmse_train_2_hyper_dt = mlModel(
        clf,
        X_train_2,
        X_test_2,
        y_train_2,
        y_test_2)
    print("DT Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_dt))
    print("DT Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_dt))
    print("DT Learning Time Dataset 2:", learn_tm_2_dt)
    print("DT Query Time Dataset 2:", query_tm_2_dt)
    print("DT Hypertuning Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_hyper_dt))
    print("DT Hypertuning Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_hyper_dt))
    print("DT Hypertuning RMSE Dataset 2: Out Sample", rmse_2_hyper_dt)
    print("DT Hypertuning RMSE Dataset 2: In Sample", rmse_train_2_hyper_dt)
    print("DT Hypertuning Learning Time Dataset 2:", learn_tm_2_hyper_dt)
    print("DT Hypertuning Query Time Dataset 2:", query_tm_2_hyper_dt)
    print("--------------------------------")
    print("--------------------------------")


'''
Result:

Grid Search for dataset car
{'criterion': 'entropy', 'max_depth': 7} DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=99) 0.740303830458489
Grid Search for dataset adult
{'criterion': 'gini', 'max_depth': 7} DecisionTreeClassifier(max_depth=7, random_state=99) 0.8374431338096457
--------------------------------
--------------------------------
DT Accuracy Dataset 1: Out Sample 0.7630057803468208
DT Accuracy Dataset 1: In Sample 0.8081058726220016
DT Learning Time Dataset 1: 0.0
DT Query Time Dataset 1: 0.015632152557373047
DT Hypertuning Accuracy Dataset 1: Out Sample 0.7842003853564548
DT Hypertuning Accuracy Dataset 1: In Sample 0.7990074441687345
DT Hypertuning RMSE Dataset 1: Out Sample 0.6898660895999755
DT Hypertuning RMSE Dataset 1: In Sample 0.6962026268273651
DT Hypertuning Learning Time Dataset 1: 0.0
DT Hypertuning Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
DT Accuracy Dataset 2: Out Sample 0.820247722387143
DT Accuracy Dataset 2: In Sample 0.9013250263250263
DT Learning Time Dataset 2: 0.21630048751831055
DT Query Time Dataset 2: 0.0
DT Hypertuning Accuracy Dataset 2: Out Sample 0.8343740403316614
DT Hypertuning Accuracy Dataset 2: In Sample 0.8426640926640927
DT Hypertuning RMSE Dataset 2: Out Sample 0.40697169393993315
DT Hypertuning RMSE Dataset 2: In Sample 0.3966559054595145
DT Hypertuning Learning Time Dataset 2: 0.10306334495544434
DT Hypertuning Query Time Dataset 2: 0.01038670539855957
--------------------------------
--------------------------------
'''