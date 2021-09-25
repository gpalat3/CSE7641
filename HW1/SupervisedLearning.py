import pandas as pd
import numpy as np
import tensorflow
import math
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

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

def runDTCurves(depth, X_train, y_train, dataset):
    print('Calculating Learning Curve Accuracy for ' + dataset)
    train_sizes, train_scores_lc, test_scores_lc, fit_times_lc, _ = learning_curve(
        DecisionTreeClassifier(random_state=99), X_train, y_train, cv=5,
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
        DecisionTreeClassifier(random_state=99), X_train, y_train, cv=5,
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

def kerasMLP():
    model = Sequential()
    model.add(Dense(units=input_units, input_shape=(input_units,), activation='relu'))
    model.add(Dense(units=hidden_layers, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def plotCurves(train_scores, test_scores, label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(train_scores, 'o-', label=label1, color='blue')
    plt.plot(test_scores, 'o-', label=label2, color='green')
    plt.legend(loc='best')
    plt.savefig(savefile)

def globalVar(X_train):
    global input_units
    global hidden_layers
    input_units = X_train.shape[1]
    hidden_layers = 100

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
    clf_DT = DecisionTreeClassifier(random_state=99)
    y_pred_1_dt, y_pred_train_1_dt, learn_tm_1_dt, query_tm_1_dt, query_tm_train_1_dt, rmse_1_dt, rmse_train_1_dt = mlModel(clf_DT,
                                                                                                       X_train_1,
                                                                                                       X_test_1,
                                                                                                       y_train_1,
                                                                                                       y_test_1)
    y_pred_2_dt, y_pred_train_2_dt, learn_tm_2_dt, query_tm_2_dt, query_tm_train_2_dt, rmse_2_dt, rmse_train_2_dt = mlModel(clf_DT,
                                                                                                       X_train_2,
                                                                                                       X_test_2,
                                                                                                       y_train_2,
                                                                                                       y_test_2)
    depth = range(1, 36)
    runDTCurves(depth, X_train_1, y_train_1, dataset='car')
    runDTCurves(depth, X_train_2, y_train_2, dataset='adult')
    grid_param = {'max_depth': depth, 'criterion': ['gini', 'entropy']}
    grid_best_params_car, grid_best_estimator_car, grid_best_score_car = gridSearch(grid_param, clf_DT, X_train_1, y_train_1, scoring='accuracy', dataset='car')
    print(grid_best_params_car, grid_best_estimator_car, grid_best_score_car)
    grid_best_params_adult, grid_best_estimator_adult, grid_best_score_adult = gridSearch(grid_param, clf_DT, X_train_2, y_train_2, scoring='accuracy', dataset='adult')
    print(grid_best_params_adult, grid_best_estimator_adult, grid_best_score_adult)
    best_params_car = {'max_depth': grid_best_params_car['max_depth'], 'criterion': grid_best_params_car['criterion']}
    best_params_adult = {'max_depth': grid_best_params_adult['max_depth'], 'criterion': grid_best_params_adult['criterion']}
    print(best_params_car)
    print(best_params_adult)
    #clf_DT = DecisionTreeClassifier(max_depth=grid_best_params_car['max_depth'], criterion=grid_best_params_car['criterion'])
    clf_DT = DecisionTreeClassifier(max_depth=grid_best_params_car['max_depth'],
                                    criterion='gini')
    y_pred_1_hyper_dt, y_pred_train_1_hyper_dt, learn_tm_1_hyper_dt, query_tm_1_hyper_dt, query_tm_train_1_hyper_dt, rmse_1_hyper_dt, rmse_train_1_hyper_dt = mlModel(
        clf_DT,
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
    y_pred_2_hyper_dt, y_pred_train_2_hyper_dt, learn_tm_2_hyper_dt, query_tm_2_hyper_dt, query_tm_train_2_hyper_dt, rmse_2_hyper_dt, rmse_train_2_hyper_dt = mlModel(
        clf_DT,
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
