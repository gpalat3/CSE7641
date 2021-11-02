import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn import metrics
pd.options.mode.chained_assignment = None

def loadAdultData(filename, col_names):
    df = pd.read_csv(filename, encoding='utf-8', header=None, names=col_names)
    df = df.dropna()
    #features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
    #            'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    features = ['workclass', 'education', 'education_num', 'occupation',
                'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    # label = ['label']
    X = pd.get_dummies(df[features])
    # y = df[label]
    y = df.iloc[:, -1]
    y.replace(' <=50K', 0, inplace=True)
    y.replace(' >50K', 1, inplace=True)
    return X.values, y.values

def loadCarData(filename, col_names):
    df = pd.read_csv(filename, encoding='utf-8', header=None, names=col_names)
    df = df.dropna()
    #features = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety']
    features = ['buying_cost', 'maintenance_cost', 'lug_boot', 'safety']
    # label = ['class']
    X = pd.get_dummies(df[features])
    # y = df[label]
    y = df.iloc[:, -1]
    y.replace('unacc', 1, inplace=True)
    y.replace('acc', 2, inplace=True)
    y.replace('good', 3, inplace=True)
    y.replace('vgood', 4, inplace=True)
    return X.values, y.values

def grpFunc(no_features, X, y, dataset, random_seed):
    fit_time = []
    rmse = []
    for i in no_features:
        print('Feature No.: ', i)
        grp = GaussianRandomProjection(n_components=i, random_state=random_seed)
        fit_start_tm = time.time()
        grp.fit(X)
        fit_end_tm = time.time()
        fit_tm = fit_end_tm - fit_start_tm
        fit_time.append(fit_tm)
        X_new = np.dot(grp.transform(X), np.linalg.pinv(np.transpose(grp.components_)))
        X_new = pd.DataFrame(data=X_new, columns=X.columns)
        rmse.append(np.sqrt(metrics.mean_squared_error(X, X_new)))
    title, xlabel, ylabel = ['GRP - ' + dataset, 'Number Of Components', 'RMSE']
    savefile = 'plots/GRP_Kurtosis_' + dataset + '.png'
    print("Plotting % for dataset % ", title, dataset)
    plotIndCurves(no_features, rmse, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['GRP Fit Times - ' + dataset, 'Number Of Components', 'Fit Time']
    savefile = 'plots/GRP_fit_times_' + dataset + '.png'
    print("Plotting % for dataset % ", title, dataset)
    plotIndCurves(no_features, fit_time, title, xlabel, ylabel, savefile)

def plotIndCurves(x, y, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, 'o-')
    plt.savefig(savefile)

if __name__ == '__main__':
    dataset_1 = 'car'
    dataset_2 = 'adult'
    col_names_car = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    X_1, y_1 = loadCarData('data/car_evaluation/car_data.csv', col_names_car)
    col_names_adult = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                       'native_country', 'label']
    X_2, y_2 = loadAdultData('data/adult/adult_data.csv', col_names_adult)
    random_seed = 99
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.3,
                                                                random_state=random_seed)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.3,
                                                                random_state=random_seed)
    scale = MinMaxScaler()
    X_1_scaled = scale.fit_transform(X_1)
    X_2_scaled = scale.fit_transform(X_2)
    no_features = np.arange(1, 15)
    grpFunc(no_features, X_1_scaled, y_1, dataset_1, random_seed)
    no_features = np.arange(1, 51)
    grpFunc(no_features, X_2_scaled, y_2, dataset_2, random_seed)
