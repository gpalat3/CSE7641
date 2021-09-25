import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import tensorflow
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
    print('Started grid Search for dataset ' + dataset)
    grid_search.fit(X, y)
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

def getMLPScores(clf, X_train, X_test, y_train, y_test, epoch):
    train_losses = []
    train_scores = []
    test_scores = []
    for i in range(epoch):
        clf.fit(X_train, y_train.values.ravel())
        y_pred_train = clf.predict(X_train)
        y_pred = clf.predict(X_test)
        acc_train = metrics.accuracy_score(y_train, y_pred_train)
        acc_test = metrics.accuracy_score(y_test, y_pred)
        train_losses.append(clf.loss_)
        train_scores.append(acc_train)
        test_scores.append(acc_test)
    return train_losses, train_scores, test_scores

def kerasMLP():
    model = Sequential()
    model.add(Dense(units=input_units, input_shape=(input_units,), activation='relu'))
    model.add(Dense(units=hidden_layers, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def plotMLPLossCurve(epoch, train_losses, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(np.arange(1, epoch + 1), train_losses, label='Training Loss')
    plt.legend(loc='best')
    plt.savefig(savefile)

def plotCurves(train_scores, test_scores, label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(train_scores, 'o-', label=label1, color='blue')
    plt.plot(test_scores, 'o-', label=label2, color='green')
    plt.legend(loc='best')
    plt.savefig(savefile)

def plotMLPValidationCurve(epoch, train_scores, test_scores, label1, label2, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(np.arange(1, epoch + 1), train_scores, 'o-', label=label1, color='blue')
    plt.plot(np.arange(1, epoch + 1), test_scores, 'o-', label=label2, color='green')
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
    '''
    clf_MLP = MLPClassifier(random_state=99, max_iter=1, warm_start=True)
    y_pred_1_mlp, y_pred_train_1_mlp, learn_tm_1_mlp, query_tm_1_mlp, query_tm_train_1_mlp, rmse_1_mlp, rmse_train_1_mlp = mlModel(clf_MLP, X_train_1, X_test_1,
                                                                                       y_train_1, y_test_1)
    print("--------------------------------")
    print("--------------------------------")
    print("MLP Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_mlp))
    print("MLP Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_mlp))
    print("MLP RMSE Dataset 1: Out Sample", rmse_1_mlp)
    print("MLP RMSE Dataset 1: In Sample", rmse_train_1_mlp)
    print("MLP Learning Time Dataset 1:", learn_tm_1_mlp)
    print("MLP Query Time Dataset 1:", query_tm_1_mlp)
    print("--------------------------------")
    print("--------------------------------")
    '''
    '''
    epoch = 400
    dataset = 'car'
    savefile = 'plots/MLP_Loss_Curve_Scikit_Learn_' + dataset + '.png'
    title, xlabel, ylabel = ['Loss Curve', 'Epoch', 'Accuracy']
    train_losses, train_scores, test_scores = getMLPScores(clf_MLP, X_train_1, X_test_1, y_train_1, y_test_1, epoch)
    plotMLPLossCurve(epoch, train_losses, title, xlabel, ylabel, savefile)
    '''
    input_units = 10
    hidden_layers = 10
    globalVar(X_train_1)
    epochs = 200
    batch_size = 100
    clf = KerasClassifier(build_fn=kerasMLP, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=0)
    history_1 = clf.fit(X_train_1, y_train_1)
    y_pred_1_mlp = clf.predict(X_test_1)
    y_pred_train_1_mlp = clf.predict(X_train_1)
    train_losses_1 = history_1.history['loss']
    train_acc_1 = history_1.history['accuracy']
    val_losses_1 = history_1.history['val_loss']
    val_acc_1 = history_1.history['val_accuracy']
    dataset = 'car'
    savefile = 'plots/MLP_Loss_Curve_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Loss', 'Cross Validation Loss', 'Loss Curve',
                                             'Epoch', 'Loss']
    print('Plotting Loss Curve for dataset 1 ' + dataset)
    plotCurves(train_losses_1, val_losses_1, label1, label2, title, xlabel, ylabel, savefile)
    savefile = 'plots/MLP_Validation_Curve_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Validation Curve',
                                             'Epoch', 'Accuracy']
    print('Plotting Validation Curve for dataset 1 ' + dataset)
    plotCurves(train_acc_1, val_acc_1, label1, label2, title, xlabel, ylabel, savefile)
    globalVar(X_train_2)
    clf = KerasClassifier(build_fn=kerasMLP, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=0)
    history_2 = clf.fit(X_train_2, y_train_2)
    y_pred_2_mlp = clf.predict(X_test_2)
    y_pred_train_2_mlp = clf.predict(X_train_2)
    train_losses_2 = history_2.history['loss']
    train_acc_2 = history_2.history['accuracy']
    val_losses_2 = history_2.history['val_loss']
    val_acc_2 = history_2.history['val_accuracy']
    dataset = 'adult'
    savefile = 'plots/MLP_Loss_Curve_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Loss', 'Cross Validation Loss', 'Loss Curve',
                                             'Epoch', 'Loss']
    print('Plotting Loss Curve for dataset 2 ' + dataset)
    plotCurves(train_losses_2, val_losses_2, label1, label2, title, xlabel, ylabel, savefile)
    savefile = 'plots/MLP_Validation_Curve_' + dataset + '.png'
    label1, label2, title, xlabel, ylabel = ['Training Score', 'Cross Validation Score', 'Validation Curve',
                                             'Epoch', 'Accuracy']
    print('Plotting Validation Curve for dataset 2 ' + dataset)
    plotCurves(train_acc_2, val_acc_2, label1, label2, title, xlabel, ylabel, savefile)
    globalVar(X_train_1)
    grid_param = {'epochs': [1, 10, 20, 30, 50], 'batch_size': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    clf = KerasClassifier(build_fn=kerasMLP, validation_split=0.3, verbose=0)
    grid_best_params_1, grid_best_estimator_1, grid_best_score_1 = gridSearch(grid_param, clf, X_train_1,
                                                                              y_train_1, scoring='accuracy',
                                                                              dataset='car')
    print('Grid search results for dataset 1: ', grid_best_params_1)
    clf = KerasClassifier(build_fn=kerasMLP, epochs=grid_best_params_1['epochs'], batch_size=grid_best_params_1['batch_size'], validation_split=0.3, verbose=0)
    clf.fit(X_train_1, y_train_1)
    y_pred_1_hyper_mlp = clf.predict(X_test_1)
    y_pred_train_1_hyper_mlp = clf.predict(X_train_1)
    print("--------------------------------")
    print("--------------------------------")
    print("MLP Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_mlp))
    print("MLP Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_mlp))
    print("MLP Hypertuning Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_1_hyper_mlp))
    print("MLP Hypertuning Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_1_hyper_mlp))
    print("--------------------------------")
    print("--------------------------------")
    globalVar(X_train_2)
    clf = KerasClassifier(build_fn=kerasMLP, validation_split=0.3, verbose=0)
    grid_best_params_2, grid_best_estimator_2, grid_best_score_2 = gridSearch(grid_param, clf, X_train_2,
                                                                              y_train_2, scoring='accuracy',
                                                                              dataset='adult')
    print('Grid search results for dataset 2: ', grid_best_params_2)
    clf = KerasClassifier(build_fn=kerasMLP, epochs=grid_best_params_2['epochs'], batch_size=grid_best_params_2['batch_size'],
                          validation_split=0.3, verbose=0)
    clf.fit(X_train_2, y_train_2)
    y_pred_2_hyper_mlp = clf.predict(X_test_2)
    y_pred_train_2_hyper_mlp = clf.predict(X_train_2)
    print("--------------------------------")
    print("--------------------------------")
    print("MLP Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_mlp))
    print("MLP Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_mlp))
    print("MLP Hypertuning Accuracy Dataset 2: Out Sample", metrics.accuracy_score(y_test_2, y_pred_2_hyper_mlp))
    print("MLP Hypertuning Accuracy Dataset 2: In Sample", metrics.accuracy_score(y_train_2, y_pred_train_2_hyper_mlp))
    print("--------------------------------")
    print("--------------------------------")

'''
Results:

Grid search results for dataset 1:  {'epochs': 20}
--------------------------------
--------------------------------
MLP Accuracy Dataset 1: Out Sample 0.7360308285163777
MLP Accuracy Dataset 1: In Sample 0.7303556658395368
MLP Hypertuning Accuracy Dataset 1: Out Sample 0.7418111753371869
MLP Hypertuning Accuracy Dataset 1: In Sample 0.7237386269644334
--------------------------------
--------------------------------
Grid search results for dataset 2:  {'epochs': 30}
--------------------------------
--------------------------------
MLP Accuracy Dataset 2: Out Sample 0.7538130821987921
MLP Accuracy Dataset 2: In Sample 0.795015795015795
MLP Hypertuning Accuracy Dataset 2: Out Sample 0.7799160610093152
MLP Hypertuning Accuracy Dataset 2: In Sample 0.7813706563706564
--------------------------------
--------------------------------
'''
