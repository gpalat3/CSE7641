import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import mlrose_hiive
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
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

def mlModel(clf, X_train, X_test, y_train, y_test):
    learn_start_tm = time.time()
    clf.fit(X_train, y_train)
    learn_end_tm = time.time()
    learn_tm = learn_end_tm - learn_start_tm
    curves = clf.fitness_curve
    query_start_tm = time.time()
    y_pred = clf.predict(X_test)
    query_end_tm = time.time()
    query_tm = query_end_tm - query_start_tm
    query_start_tm_train = time.time()
    y_pred_train = clf.predict(X_train)
    query_end_tm_train = time.time()
    query_tm_train = query_end_tm_train - query_start_tm_train
    rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    rmse_train = math.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
    y_train_acc = metrics.accuracy_score(y_train, y_pred_train)
    y_test_acc =metrics.accuracy_score(y_test, y_pred)
    return y_pred, y_pred_train, learn_tm, query_tm, query_tm_train, rmse, rmse_train, curves, y_train_acc, y_test_acc

def plotCurves(iterations, curves_rhc, curves_sa, curves_ga, curves_backprop, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(iterations, curves_rhc, label='RHC')
    plt.plot(iterations, curves_sa, label='SA')
    plt.plot(iterations, curves_ga, label='GA')
    plt.plot(iterations, curves_backprop, label='BackProp')
    plt.legend(loc='best')
    plt.savefig(savefile)

def plot3ROCurves(iterations, curves_rhc, curves_sa, curves_ga, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(iterations, curves_rhc, label='RHC')
    plt.plot(iterations, curves_sa, label='SA')
    plt.plot(iterations, curves_ga, label='GA')
    plt.legend(loc='best')
    plt.savefig(savefile)

def plotIndCurves(iterations, curves, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(iterations, curves, label='RHC')
    plt.legend(loc='best')
    plt.savefig(savefile)

if __name__ == '__main__':
    col_names_car = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    X_1, y_1 = loadCarData('data/car_evaluation/car_data.csv', col_names_car)
    col_names_adult = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label']
    X_2, y_2 = loadAdultData('data/adult/adult_data.csv', col_names_adult)
    random_seed = 99
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.3,
                                                                        random_state=random_seed)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.3,
                                                                        random_state=random_seed)
    scale = MinMaxScaler()
    hot = OneHotEncoder()
    X_train_1_scaled = scale.fit_transform(X_train_1)
    X_test_1_scaled = scale.fit_transform(X_test_1)
    y_train_1_hot = hot.fit_transform(y_train_1.values.reshape(-1, 1)).todense()
    y_test_1_hot = hot.fit_transform(y_test_1.values.reshape(-1, 1)).todense()
    X_train_2_scaled = scale.fit_transform(X_train_2)
    X_test_2_scaled = scale.fit_transform(X_test_2)
    y_train_2_hot = hot.fit_transform(y_train_2.values.reshape(-1, 1)).todense()
    y_test_2_hot = hot.fit_transform(y_test_2.values.reshape(-1, 1)).todense()
    clf_rhc = mlrose_hiive.NeuralNetwork(random_state=random_seed, algorithm='random_hill_climb', hidden_nodes=[10], activation='relu',
                                     learning_rate=0.0001, max_attempts=100, max_iters=1000, clip_max=5,
                                     is_classifier=True, curve=True)
    clf_rhc_5 = mlrose_hiive.NeuralNetwork(random_state=random_seed, algorithm='random_hill_climb', hidden_nodes=[10],
                                         activation='relu',
                                         learning_rate=0.0001, max_attempts=100, max_iters=1000, clip_max=5, restarts=5,
                                         is_classifier=True, curve=True)
    clf_rhc_10 = mlrose_hiive.NeuralNetwork(random_state=random_seed, algorithm='random_hill_climb', hidden_nodes=[10],
                                         activation='relu',
                                         learning_rate=0.0001, max_attempts=100, max_iters=1000, clip_max=5, restarts=10,
                                         is_classifier=True, curve=True)
    clf_rhc_15 = mlrose_hiive.NeuralNetwork(random_state=random_seed, algorithm='random_hill_climb', hidden_nodes=[10],
                                         activation='relu',
                                         learning_rate=0.0001, max_attempts=100, max_iters=1000, clip_max=5, restarts=15,
                                         is_classifier=True, curve=True)
    clf_sa = mlrose_hiive.NeuralNetwork(random_state=random_seed, algorithm='simulated_annealing', hidden_nodes=[10], activation='relu',
                                     learning_rate=0.0001, max_attempts=100, max_iters=1000, clip_max=5,
                                     is_classifier=True, curve=True)
    clf_ga = mlrose_hiive.NeuralNetwork(random_state=random_seed, algorithm='genetic_alg', hidden_nodes=[10], activation='relu',
                                     learning_rate=0.0001, max_attempts=100, max_iters=1000, clip_max=5,
                                     is_classifier=True, curve=True)
    clf_backprop = mlrose_hiive.NeuralNetwork(random_state=random_seed, algorithm='gradient_descent', hidden_nodes=[10],
                                        activation='relu',
                                        learning_rate=0.0001, max_attempts=100, max_iters=1000,
                                        clip_max=5,
                                        is_classifier=True, curve=True)
    y_pred_1_rhc, y_pred_train_1_rhc, learn_tm_1_rhc, query_tm_1_rhc, query_tm_train_1_rhc, rmse_1_rhc, rmse_train_1_rhc, curves_1_rhc, y_train_acc_1_rhc, y_test_acc_1_rhc = mlModel(
        clf_rhc, X_train_1_scaled, X_test_1_scaled, y_train_1_hot, y_test_1_hot)
    print("--------------------------------")
    print("--------------------------------")
    print("NN RHC Accuracy Dataset 1: Out Sample", y_test_acc_1_rhc)
    print("NN RHC Accuracy Dataset 1: In Sample", y_train_acc_1_rhc)
    print("NN RHC RMSE Dataset 1: Out Sample", rmse_1_rhc)
    print("NN RHC RMSE Dataset 1: In Sample", rmse_train_1_rhc)
    print("NN RHC Learning Time Dataset 1:", learn_tm_1_rhc)
    print("NN RHC Query Time Dataset 1:", query_tm_1_rhc)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_1_rhc_5, y_pred_train_1_rhc_5, learn_tm_1_rhc_5, query_tm_1_rhc_5, query_tm_train_1_rhc_5, rmse_1_rhc_5, rmse_train_1_rhc_5, curves_1_rhc_5, y_train_acc_1_rhc_5, y_test_acc_1_rhc_5 = mlModel(
        clf_rhc_5, X_train_1_scaled, X_test_1_scaled, y_train_1_hot, y_test_1_hot)
    print("--------------------------------")
    print("--------------------------------")
    print("NN RHC 5 Restarts Accuracy Dataset 1: Out Sample", y_test_acc_1_rhc_5)
    print("NN RHC 5 Restarts Accuracy Dataset 1: In Sample", y_train_acc_1_rhc_5)
    print("NN RHC 5 Restarts RMSE Dataset 1: Out Sample", rmse_1_rhc_5)
    print("NN RHC 5 Restarts RMSE Dataset 1: In Sample", rmse_train_1_rhc_5)
    print("NN RHC 5 Restarts Learning Time Dataset 1:", learn_tm_1_rhc_5)
    print("NN RHC 5 Restarts Query Time Dataset 1:", query_tm_1_rhc_5)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_1_rhc_10, y_pred_train_1_rhc_10, learn_tm_1_rhc_10, query_tm_1_rhc_10, query_tm_train_1_rhc_10, rmse_1_rhc_10, rmse_train_1_rhc_10, curves_1_rhc_10, y_train_acc_1_rhc_10, y_test_acc_1_rhc_10 = mlModel(
        clf_rhc_10, X_train_1_scaled, X_test_1_scaled, y_train_1_hot, y_test_1_hot)
    print("--------------------------------")
    print("--------------------------------")
    print("NN RHC 10 Restarts Accuracy Dataset 1: Out Sample", y_test_acc_1_rhc_10)
    print("NN RHC 10 Restarts Accuracy Dataset 1: In Sample", y_train_acc_1_rhc_10)
    print("NN RHC 10 Restarts RMSE Dataset 1: Out Sample", rmse_1_rhc_10)
    print("NN RHC 10 Restarts RMSE Dataset 1: In Sample", rmse_train_1_rhc_10)
    print("NN RHC 10 Restarts Learning Time Dataset 1:", learn_tm_1_rhc_10)
    print("NN RHC 10 Restarts Query Time Dataset 1:", query_tm_1_rhc_10)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_1_rhc_15, y_pred_train_1_rhc_15, learn_tm_1_rhc_15, query_tm_1_rhc_15, query_tm_train_1_rhc_15, rmse_1_rhc_15, rmse_train_1_rhc_15, curves_1_rhc_15, y_train_acc_1_rhc_15, y_test_acc_1_rhc_15 = mlModel(
        clf_rhc_15, X_train_1_scaled, X_test_1_scaled, y_train_1_hot, y_test_1_hot)
    print("--------------------------------")
    print("--------------------------------")
    print("NN RHC 15 Restarts Accuracy Dataset 1: Out Sample", y_test_acc_1_rhc_15)
    print("NN RHC 15 Restarts Accuracy Dataset 1: In Sample", y_train_acc_1_rhc_15)
    print("NN RHC 15 Restarts RMSE Dataset 1: Out Sample", rmse_1_rhc_15)
    print("NN RHC 15 Restarts RMSE Dataset 1: In Sample", rmse_train_1_rhc_15)
    print("NN RHC 15 Restarts Learning Time Dataset 1:", learn_tm_1_rhc_15)
    print("NN RHC 15 Restarts Query Time Dataset 1:", query_tm_1_rhc_15)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_1_sa, y_pred_train_1_sa, learn_tm_1_sa, query_tm_1_sa, query_tm_train_1_sa, rmse_1_sa, rmse_train_1_sa, curves_1_sa, y_train_acc_1_sa, y_test_acc_1_sa = mlModel(
        clf_sa, X_train_1_scaled, X_test_1_scaled, y_train_1_hot, y_test_1_hot)
    print("NN SA Accuracy Dataset 1: Out Sample", y_test_acc_1_sa)
    print("NN SA Accuracy Dataset 1: In Sample", y_train_acc_1_sa)
    print("NN SA RMSE Dataset 1: Out Sample", rmse_1_sa)
    print("NN SA RMSE Dataset 1: In Sample", rmse_train_1_sa)
    print("NN SA Learning Time Dataset 1:", learn_tm_1_sa)
    print("NN SA Query Time Dataset 1:", query_tm_1_sa)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_1_ga, y_pred_train_1_ga, learn_tm_1_ga, query_tm_1_ga, query_tm_train_1_ga, rmse_1_ga, rmse_train_1_ga, curves_1_ga, y_train_acc_1_ga, y_test_acc_1_ga = mlModel(
        clf_ga, X_train_1_scaled, X_test_1_scaled, y_train_1_hot, y_test_1_hot)
    print("NN GA Accuracy Dataset 1: Out Sample", y_test_acc_1_ga)
    print("NN GA Accuracy Dataset 1: In Sample", y_train_acc_1_ga)
    print("NN GA RMSE Dataset 1: Out Sample", rmse_1_ga)
    print("NN GA RMSE Dataset 1: In Sample", rmse_train_1_ga)
    print("NN GA Learning Time Dataset 1:", learn_tm_1_ga)
    print("NN GA Query Time Dataset 1:", query_tm_1_ga)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_1_backprop, y_pred_train_1_backprop, learn_tm_1_backprop, query_tm_1_backprop, query_tm_train_1_backprop, rmse_1_backprop, rmse_train_1_backprop, curves_1_backprop, y_train_acc_1_backprop, y_test_acc_1_backprop = mlModel(
        clf_backprop, X_train_1_scaled, X_test_1_scaled, y_train_1_hot, y_test_1_hot)
    print("NN BackProp Accuracy Dataset 1: Out Sample", y_test_acc_1_backprop)
    print("NN BackProp Accuracy Dataset 1: In Sample", y_train_acc_1_backprop)
    print("NN BackProp RMSE Dataset 1: Out Sample", rmse_1_backprop)
    print("NN BackProp RMSE Dataset 1: In Sample", rmse_train_1_backprop)
    print("NN BackProp Learning Time Dataset 1:", learn_tm_1_backprop)
    print("NN BackProp Query Time Dataset 1:", query_tm_1_backprop)
    print("--------------------------------")
    print("--------------------------------")
    print('RHC Curves Shape: ', curves_1_rhc.shape)
    print('SA Curves Shape: ', curves_1_sa.shape)
    print('GA Curves Shape: ', curves_1_ga.shape)
    print('BackProp Curves Shape: ', curves_1_backprop.shape)
    print('Debug RHC Shape: ', curves_1_rhc.shape, curves_1_rhc[:, :-1].shape, curves_1_rhc[:, 0:0].shape)
    '''
    print("--------------------------------")
    print(curves_1_rhc[:, :-1])
    print("--------------------------------")
    print(curves_1_rhc[:, 1:])
    print("--------------------------------")
    '''
    title, xlabel, ylabel = ['Neural Network Optimization Dataset 1', 'Iteration', 'Fitness']
    savefile = 'plots/NeuralNetwork_car_1.png'
    iterations = range(1, 1001, 1)
    plotCurves(iterations, curves_1_rhc[:, :-1], curves_1_sa[:, :-1], curves_1_ga[:, :-1], curves_1_backprop, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Neural Network Optimization Dataset 1 With RHC Restarts = 5', 'Iteration', 'Fitness']
    savefile = 'plots/NeuralNetwork_car_1_restarts_5.png'
    iterations = range(1, 1001, 1)
    plotCurves(iterations, curves_1_rhc_5[:, :-1], curves_1_sa[:, :-1], curves_1_ga[:, :-1], curves_1_backprop, title,
               xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Neural Network Optimization Dataset 1 With RHC Restarts = 10', 'Iteration', 'Fitness']
    savefile = 'plots/NeuralNetwork_car_1_restarts_10.png'
    iterations = range(1, 1001, 1)
    plotCurves(iterations, curves_1_rhc_10[:, :-1], curves_1_sa[:, :-1], curves_1_ga[:, :-1], curves_1_backprop, title,
               xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Neural Network Optimization Dataset 1 With RHC Restarts = 15', 'Iteration', 'Fitness']
    savefile = 'plots/NeuralNetwork_car_1_restarts_15.png'
    iterations = range(1, 1001, 1)
    plotCurves(iterations, curves_1_rhc_15[:, :-1], curves_1_sa[:, :-1], curves_1_ga[:, :-1], curves_1_backprop, title,
               xlabel, ylabel, savefile)
    # plotIndCurves(iterations_rhc, curves_1_rhc, title, xlabel, ylabel, savefile)
    savefile = 'plots/NeuralNetwork_car_3.png'
    title = 'Neural Network Optimization Dataset 1'
    plotCurves(iterations, curves_1_rhc, curves_1_sa, curves_1_ga, curves_1_backprop, title,
               xlabel, ylabel, savefile)
    savefile = 'plots/NeuralNetwork_car_4.png'
    plot3ROCurves(iterations, curves_1_rhc[:, :-1], curves_1_sa[:, :-1], curves_1_ga[:, :-1], title,
               xlabel, ylabel, savefile)
    ylabel = 'FEvals'
    savefile = 'plots/NeuralNetwork_car_2.png'
    plotCurves(iterations, curves_1_rhc[:, 1:], curves_1_sa[:, 1:], curves_1_ga[:, 1:], curves_1_backprop, title,
               xlabel, ylabel, savefile)
    '''
    y_pred_2_rhc, y_pred_train_2_rhc, learn_tm_2_rhc, query_tm_2_rhc, query_tm_train_2_rhc, rmse_2_rhc, rmse_train_2_rhc, curves_2_rhc, y_train_acc_2_rhc, y_test_acc_2_rhc = mlModel(
        clf_rhc, X_train_2_scaled, X_test_2_scaled, y_train_2_hot, y_test_2_hot)
    print("--------------------------------")
    print("--------------------------------")
    print("NN RHC Accuracy Dataset 2: Out Sample", y_test_acc_2_rhc)
    print("NN RHC Accuracy Dataset 2: In Sample", y_train_acc_2_rhc)
    print("NN RHC RMSE Dataset 2: Out Sample", rmse_2_rhc)
    print("NN RHC RMSE Dataset 2: In Sample", rmse_train_2_rhc)
    print("NN RHC Learning Time Dataset 2:", learn_tm_2_rhc)
    print("NN RHC Query Time Dataset 2:", query_tm_2_rhc)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_2_sa, y_pred_train_2_sa, learn_tm_2_sa, query_tm_2_sa, query_tm_train_2_sa, rmse_2_sa, rmse_train_2_sa, curves_2_sa, y_train_acc_2_sa, y_test_acc_2_sa = mlModel(
        clf_sa, X_train_2_scaled, X_test_2_scaled, y_train_2_hot, y_test_2_hot)
    print("NN SA Accuracy Dataset 2: Out Sample", y_test_acc_2_sa)
    print("NN SA Accuracy Dataset 2: In Sample", y_train_acc_2_sa)
    print("NN SA RMSE Dataset 2: Out Sample", rmse_2_sa)
    print("NN SA RMSE Dataset 2: In Sample", rmse_train_2_sa)
    print("NN SA Learning Time Dataset 2:", learn_tm_2_sa)
    print("NN SA Query Time Dataset 2:", query_tm_2_sa)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_2_ga, y_pred_train_2_ga, learn_tm_2_ga, query_tm_2_ga, query_tm_train_2_ga, rmse_2_ga, rmse_train_2_ga, curves_2_ga, y_train_acc_2_ga, y_test_acc_2_ga = mlModel(
        clf_ga, X_train_2_scaled, X_test_2_scaled, y_train_2_hot, y_test_2_hot)
    print("NN GA Accuracy Dataset 2: Out Sample", y_test_acc_2_ga)
    print("NN GA Accuracy Dataset 2: In Sample", y_train_acc_2_ga)
    print("NN GA RMSE Dataset 2: Out Sample", rmse_2_ga)
    print("NN GA RMSE Dataset 2: In Sample", rmse_train_2_ga)
    print("NN GA Learning Time Dataset 2:", learn_tm_2_ga)
    print("NN GA Query Time Dataset 2:", query_tm_2_ga)
    print("--------------------------------")
    print("--------------------------------")
    y_pred_2_backprop, y_pred_train_2_backprop, learn_tm_2_backprop, query_tm_2_backprop, query_tm_train_2_backprop, rmse_2_backprop, rmse_train_2_backprop, curves_2_backprop, y_train_acc_2_backprop, y_test_acc_2_backprop = mlModel(
        clf_backprop, X_train_2_scaled, X_test_2_scaled, y_train_2_hot, y_test_2_hot)
    print("NN BackProp Accuracy Dataset 2: Out Sample", y_test_acc_2_backprop)
    print("NN BackProp Accuracy Dataset 2: In Sample", y_train_acc_2_backprop)
    print("NN BackProp RMSE Dataset 2: Out Sample", rmse_2_backprop)
    print("NN BackProp RMSE Dataset 2: In Sample", rmse_train_2_backprop)
    print("NN BackProp Learning Time Dataset 2:", learn_tm_2_backprop)
    print("NN BackProp Query Time Dataset 2:", query_tm_2_backprop)
    print("--------------------------------")
    print("--------------------------------")
    print('RHC Curves Shape: ', curves_2_rhc.shape)
    print('SA Curves Shape: ', curves_2_sa.shape)
    print('GA Curves Shape: ', curves_2_ga.shape)
    print('BackProp Curves Shape: ', curves_2_backprop.shape)
    print('Debug RHC Shape: ', curves_2_rhc.shape, curves_2_rhc[:, :-1].shape, curves_2_rhc[:, 0:0].shape)
    title, xlabel, ylabel = ['Neural Network Optimization Dataset 2', 'Iteration', 'Fitness']
    savefile = 'plots/NeuralNetwork_adult_1.png'
    iterations = range(1, 1001, 1)
    plotCurves(iterations, curves_2_rhc[:, :-1], curves_2_sa[:, :-1], curves_2_ga[:, :-1], curves_2_backprop, title,
               xlabel, ylabel, savefile)
    savefile = 'plots/NeuralNetwork_adult_3.png'
    plotCurves(iterations, curves_2_rhc, curves_2_sa, curves_2_ga, curves_2_backprop, title,
               xlabel, ylabel, savefile)
    savefile = 'plots/NeuralNetwork_adult_4.png'
    plot3ROCurves(iterations, curves_2_rhc[:, :-1], curves_2_sa[:, :-1], curves_2_ga[:, :-1], title,
                  xlabel, ylabel, savefile)
    ylabel = 'FEvals'
    savefile = 'plots/NeuralNetwork_adult_2.png'
    plotCurves(iterations, curves_2_rhc[:, 1:], curves_2_sa[:, 1:], curves_2_ga[:, 1:], curves_2_backprop, title,
               xlabel, ylabel, savefile)
    '''

'''
Dataset 1:
--------------------------------
--------------------------------
NN RHC Accuracy Dataset 1: Out Sample 0.07707129094412331
NN RHC Accuracy Dataset 1: In Sample 0.08188585607940446
NN RHC RMSE Dataset 1: Out Sample 0.6793116770142689
NN RHC RMSE Dataset 1: In Sample 0.6775375059436177
NN RHC Learning Time Dataset 1: 2.6160521507263184
NN RHC Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
NN SA Accuracy Dataset 1: Out Sample 0.07707129094412331
NN SA Accuracy Dataset 1: In Sample 0.08188585607940446
NN SA RMSE Dataset 1: Out Sample 0.6793116770142689
NN SA RMSE Dataset 1: In Sample 0.6775375059436176
NN SA Learning Time Dataset 1: 4.163290977478027
NN SA Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
NN GA Accuracy Dataset 1: Out Sample 0.6936416184971098
NN GA Accuracy Dataset 1: In Sample 0.7030603804797353
NN GA RMSE Dataset 1: Out Sample 0.3913811323396225
NN GA RMSE Dataset 1: In Sample 0.3853178035857315
NN GA Learning Time Dataset 1: 360.12134647369385
NN GA Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
NN BackProp Accuracy Dataset 1: Out Sample 0.766859344894027
NN BackProp Accuracy Dataset 1: In Sample 0.7857733664185277
NN BackProp RMSE Dataset 1: Out Sample 0.34142397038431044
NN BackProp RMSE Dataset 1: In Sample 0.3272817086100843
NN BackProp Learning Time Dataset 1: 3.6578235626220703
NN BackProp Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
RHC Curves Shape:  (1000, 2)
SA Curves Shape:  (1000, 2)
GA Curves Shape:  (1000, 2)
BackProp Curves Shape:  (1000,)
Debug RHC Shape:  (1000, 2) (1000, 1) (1000, 0)
--------------------------------

with 200 epochs
MLP Accuracy Dataset 1: Out Sample 0.7360308285163777
MLP Accuracy Dataset 1: In Sample 0.7303556658395368
with 10 epochs
MLP Accuracy Dataset 1: Out Sample 0.7090558766859345
MLP Accuracy Dataset 1: In Sample 0.6997518610421837
MLP Hypertuning Accuracy Dataset 1: Out Sample 0.7418111753371869
MLP Hypertuning Accuracy Dataset 1: In Sample 0.7303556658395368
--------------------------------
--------------------------------









Dataset 1:
C:\ProgramData\Anaconda3\envs\HW2\python.exe D:/OMSCS/ML/HW2/NeuralNetwork.py
--------------------------------
--------------------------------
NN RHC Accuracy Dataset 1: Out Sample 0.07707129094412331
NN RHC Accuracy Dataset 1: In Sample 0.08188585607940446
NN RHC RMSE Dataset 1: Out Sample 0.6793116770142689
NN RHC RMSE Dataset 1: In Sample 0.6775375059436177
NN RHC Learning Time Dataset 1: 3.231224536895752
NN RHC Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
NN SA Accuracy Dataset 1: Out Sample 0.07707129094412331
NN SA Accuracy Dataset 1: In Sample 0.08188585607940446
NN SA RMSE Dataset 1: Out Sample 0.6793116770142689
NN SA RMSE Dataset 1: In Sample 0.6775375059436176
NN SA Learning Time Dataset 1: 4.78082537651062
NN SA Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
NN GA Accuracy Dataset 1: Out Sample 0.6936416184971098
NN GA Accuracy Dataset 1: In Sample 0.7030603804797353
NN GA RMSE Dataset 1: Out Sample 0.3913811323396225
NN GA RMSE Dataset 1: In Sample 0.3853178035857315
NN GA Learning Time Dataset 1: 375.4478003978729
NN GA Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
NN BackProp Accuracy Dataset 1: Out Sample 0.766859344894027
NN BackProp Accuracy Dataset 1: In Sample 0.7857733664185277
NN BackProp RMSE Dataset 1: Out Sample 0.34142397038431044
NN BackProp RMSE Dataset 1: In Sample 0.3272817086100843
NN BackProp Learning Time Dataset 1: 3.8004634380340576
NN BackProp Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
RHC Curves Shape:  (1000, 2)
SA Curves Shape:  (1000, 2)
GA Curves Shape:  (1000, 2)
BackProp Curves Shape:  (1000,)
Debug RHC Shape:  (1000, 2) (1000, 1) (1000, 0)
--------------------------------
--------------------------------
NN RHC Accuracy Dataset 2: Out Sample 0.636298495240045
NN RHC Accuracy Dataset 2: In Sample 0.6418918918918919
NN RHC RMSE Dataset 2: Out Sample 0.6030766989031784
NN RHC RMSE Dataset 2: In Sample 0.598421346634717
NN RHC Learning Time Dataset 2: 41.54875588417053
NN RHC Query Time Dataset 2: 0.008362054824829102
--------------------------------
--------------------------------
NN SA Accuracy Dataset 2: Out Sample 0.636298495240045
NN SA Accuracy Dataset 2: In Sample 0.6419357669357669
NN SA RMSE Dataset 2: Out Sample 0.6030766989031784
NN SA RMSE Dataset 2: In Sample 0.5983846865221678
NN SA Learning Time Dataset 2: 52.05929231643677
NN SA Query Time Dataset 2: 0.00700068473815918
--------------------------------
--------------------------------
NN GA Accuracy Dataset 2: Out Sample 0.7850342921486334
NN GA Accuracy Dataset 2: In Sample 0.7836960336960337
NN GA RMSE Dataset 2: Out Sample 0.4636439451253155
NN GA RMSE Dataset 2: In Sample 0.46508490225330507
NN GA Learning Time Dataset 2: 5534.499252080917
NN GA Query Time Dataset 2: 0.012999773025512695
--------------------------------
--------------------------------
NN BackProp Accuracy Dataset 2: Out Sample 0.8132869280376702
NN BackProp Accuracy Dataset 2: In Sample 0.8114250614250614
NN BackProp RMSE Dataset 2: Out Sample 0.4321030802509163
NN BackProp RMSE Dataset 2: In Sample 0.4342521601269688
NN BackProp Learning Time Dataset 2: 70.61483216285706
NN BackProp Query Time Dataset 2: 0.013007402420043945
--------------------------------
--------------------------------
RHC Curves Shape:  (1000, 2)
SA Curves Shape:  (1000, 2)
GA Curves Shape:  (1000, 2)
BackProp Curves Shape:  (1000,)
Debug RHC Shape:  (1000, 2) (1000, 1) (1000, 0)

Process finished with exit code 0

'''







'''
NN RHC Accuracy Dataset 1: Out Sample 0.07707129094412331
NN RHC Accuracy Dataset 1: In Sample 0.08188585607940446
NN RHC RMSE Dataset 1: Out Sample 0.6793116770142689
NN RHC RMSE Dataset 1: In Sample 0.6775375059436177
NN RHC Learning Time Dataset 1: 3.1111645698547363
NN RHC Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
--------------------------------
--------------------------------
NN RHC 5 Restarts Accuracy Dataset 1: Out Sample 0.07707129094412331
NN RHC 5 Restarts Accuracy Dataset 1: In Sample 0.08188585607940446
NN RHC 5 Restarts RMSE Dataset 1: Out Sample 0.6793116770142689
NN RHC 5 Restarts RMSE Dataset 1: In Sample 0.6775375059436176
NN RHC 5 Restarts Learning Time Dataset 1: 18.70243239402771
NN RHC 5 Restarts Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
--------------------------------
--------------------------------
NN RHC 10 Restarts Accuracy Dataset 1: Out Sample 0.07707129094412331
NN RHC 10 Restarts Accuracy Dataset 1: In Sample 0.08188585607940446
NN RHC 10 Restarts RMSE Dataset 1: Out Sample 0.6793116770142689
NN RHC 10 Restarts RMSE Dataset 1: In Sample 0.6775375059436177
NN RHC 10 Restarts Learning Time Dataset 1: 32.088632106781006
NN RHC 10 Restarts Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
--------------------------------
--------------------------------
NN RHC 15 Restarts Accuracy Dataset 1: Out Sample 0.07707129094412331
NN RHC 15 Restarts Accuracy Dataset 1: In Sample 0.08188585607940446
NN RHC 15 Restarts RMSE Dataset 1: Out Sample 0.6793116770142689
NN RHC 15 Restarts RMSE Dataset 1: In Sample 0.6775375059436177
NN RHC 15 Restarts Learning Time Dataset 1: 45.497854471206665
NN RHC 15 Restarts Query Time Dataset 1: 0.0010166168212890625
--------------------------------
--------------------------------
NN SA Accuracy Dataset 1: Out Sample 0.07707129094412331
NN SA Accuracy Dataset 1: In Sample 0.08188585607940446
NN SA RMSE Dataset 1: Out Sample 0.6793116770142689
NN SA RMSE Dataset 1: In Sample 0.6775375059436176
NN SA Learning Time Dataset 1: 4.095730781555176
NN SA Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
NN GA Accuracy Dataset 1: Out Sample 0.6936416184971098
NN GA Accuracy Dataset 1: In Sample 0.7030603804797353
NN GA RMSE Dataset 1: Out Sample 0.3913811323396225
NN GA RMSE Dataset 1: In Sample 0.3853178035857315
NN GA Learning Time Dataset 1: 455.1071650981903
NN GA Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
NN BackProp Accuracy Dataset 1: Out Sample 0.766859344894027
NN BackProp Accuracy Dataset 1: In Sample 0.7857733664185277
NN BackProp RMSE Dataset 1: Out Sample 0.34142397038431044
NN BackProp RMSE Dataset 1: In Sample 0.3272817086100843
NN BackProp Learning Time Dataset 1: 3.7105257511138916
NN BackProp Query Time Dataset 1: 0.0
--------------------------------
--------------------------------
RHC Curves Shape:  (1000, 2)
SA Curves Shape:  (1000, 2)
GA Curves Shape:  (1000, 2)
BackProp Curves Shape:  (1000,)
Debug RHC Shape:  (1000, 2) (1000, 1) (1000, 0)

Process finished with exit code 0

'''