import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, MiniBatchDictionaryLearning
from sklearn.random_projection import GaussianRandomProjection
pd.options.mode.chained_assignment = None

def loadAdultData(filename, col_names):
    df = pd.read_csv(filename, encoding='utf-8', header=None, names=col_names)
    df = df.dropna()
    #features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
    #            'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    features = ['workclass', 'education', 'education_num', 'occupation',
                'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    X = pd.get_dummies(df[features])
    y = df.iloc[:, -1]
    y.replace(' <=50K', 0, inplace=True)
    y.replace(' >50K', 1, inplace=True)
    return X.values, y.values

def loadCarData(filename, col_names):
    df = pd.read_csv(filename, encoding='utf-8', header=None, names=col_names)
    df = df.dropna()
    #features = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety']
    features = ['buying_cost', 'maintenance_cost', 'lug_boot', 'safety']
    X = pd.get_dummies(df[features])
    y = df.iloc[:, -1]
    y.replace('unacc', 1, inplace=True)
    y.replace('acc', 2, inplace=True)
    y.replace('good', 3, inplace=True)
    y.replace('vgood', 4, inplace=True)
    return X.values, y.values

def trainTestSplit(X, y, sample_size, random_seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sample_size, random_state=random_seed)
    return X_train, X_test, y_train, y_test

def gridSearch(grid_param, clf, X, y, scoring, dataset):
    grid_search = GridSearchCV(estimator=clf, param_grid=grid_param, scoring=scoring, cv=5, n_jobs=-1,
                               return_train_score=True)
    print('Started grid Search for dataset ' + dataset)
    grid_search.fit(X, y)
    print('Ended grid Search for dataset ' + dataset)
    return grid_search.best_params_, grid_search.best_estimator_, grid_search.best_score_

def mlModel(clf, X_train, X_test, y_train, y_test):
    dt_learn_start_tm = time.time()
    clf.fit(X_train, y_train)
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

if __name__ == '__main__':
    col_names_car = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    X_1, y_1 = loadCarData('data/car_evaluation/car_data.csv', col_names_car)
    col_names_adult = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label']
    X_2, y_2 = loadAdultData('data/adult/adult_data.csv', col_names_adult)
    random_seed = 99
    dataset_1 = 'car'
    dataset_2 = 'adult'
    scale = MinMaxScaler()
    X_1_scaled = scale.fit_transform(X_1)
    X_2_scaled = scale.fit_transform(X_2)
    pca_1 = PCA(n_components=9, random_state=random_seed)
    pca_2 = PCA(n_components=30, random_state=random_seed)
    ica_1 = FastICA(n_components=7, random_state=random_seed)
    ica_2 = FastICA(n_components=10, random_state=random_seed)
    grp_1 = GaussianRandomProjection(n_components=9, random_state=random_seed)
    grp_2 = GaussianRandomProjection(n_components=10, random_state=random_seed)
    mbdl_1 = MiniBatchDictionaryLearning(n_components=21, random_state=random_seed, batch_size=200, n_iter=100, alpha=1)
    mbdl_2 = MiniBatchDictionaryLearning(n_components=4, random_state=random_seed, batch_size=200, n_iter=100, alpha=1)
    X_1_pca = pca_1.fit_transform(X_1_scaled)
    X_2_pca = pca_2.fit_transform(X_2_scaled)
    X_1_ica = ica_1.fit_transform(X_1_scaled)
    X_2_ica = ica_2.fit_transform(X_2_scaled)
    X_1_grp = grp_1.fit_transform(X_1_scaled)
    X_2_grp = grp_2.fit_transform(X_2_scaled)
    X_1_mbdl = mbdl_1.fit_transform(X_1_scaled)
    X_2_mbdl = mbdl_2.fit_transform(X_2_scaled)

    sample_size = 0.3
    clf = MLPClassifier(random_state=random_seed)
    grid_param = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                  'activation': ['tanh', 'relu'],
                  'solver': ['sgd', 'adam'],
                  'alpha': [0.0001, 0.01, 0.05], }

    '''
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = trainTestSplit(X_1_pca, y_1, sample_size, random_seed)
    grid_search_pca = gridSearch(grid_param, clf, X_train_pca, y_train_pca, 'accuracy', dataset_1)

    X_train_ica, X_test_ica, y_train_ica, y_test_ica = trainTestSplit(X_1_ica, y_1, sample_size, random_seed)
    grid_search_ica = gridSearch(grid_param, clf, X_train_ica, y_train_ica, 'accuracy', dataset_1)

    X_train_grp, X_test_grp, y_train_grp, y_test_grp = trainTestSplit(X_1_grp, y_1, sample_size, random_seed)
    grid_search_grp = gridSearch(grid_param, clf, X_train_grp, y_train_grp, 'accuracy', dataset_1)

    X_train_mbdl, X_test_mbdl, y_train_mbdl, y_test_mbdl = trainTestSplit(X_1_mbdl, y_1, sample_size, random_seed)
    grid_search_mbdl = gridSearch(grid_param, clf, X_train_mbdl, y_train_mbdl, 'accuracy', dataset_1)
    
    print('PCA Grid Search: ', grid_search_pca)
    print('ICA Grid Search: ', grid_search_ica)
    print('GRP Grid Search: ', grid_search_grp)
    print('MBDL Grid Search: ', grid_search_mbdl)
    '''

    '''
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = trainTestSplit(X_2_pca, y_2, sample_size, random_seed)
    grid_search_pca = gridSearch(grid_param, clf, X_train_pca, y_train_pca, 'accuracy', dataset_2)

    X_train_ica, X_test_ica, y_train_ica, y_test_ica = trainTestSplit(X_2_ica, y_2, sample_size, random_seed)
    grid_search_ica = gridSearch(grid_param, clf, X_train_ica, y_train_ica, 'accuracy', dataset_2)

    X_train_grp, X_test_grp, y_train_grp, y_test_grp = trainTestSplit(X_2_grp, y_2, sample_size, random_seed)
    grid_search_grp = gridSearch(grid_param, clf, X_train_grp, y_train_grp, 'accuracy', dataset_2)

    X_train_mbdl, X_test_mbdl, y_train_mbdl, y_test_mbdl = trainTestSplit(X_2_mbdl, y_2, sample_size, random_seed)
    grid_search_mbdl = gridSearch(grid_param, clf, X_train_mbdl, y_train_mbdl, 'accuracy', dataset_2)

    print('PCA Grid Search: ', grid_search_pca)
    print('ICA Grid Search: ', grid_search_ica)
    print('GRP Grid Search: ', grid_search_grp)
    print('MBDL Grid Search: ', grid_search_mbdl)
    '''
    '''
    Car:
    PCA Grid Search:  ({'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50),
              random_state=99), 0.7609787044340043)
    ICA Grid Search:  ({'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(alpha=0.01, hidden_layer_sizes=(50, 100, 50), random_state=99), 0.7096704502589075)
    GRP Grid Search:  ({'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 100, 50),
              random_state=99), 0.7626315969959878)
    MBDL Grid Search:  ({'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'sgd'}, MLPClassifier(hidden_layer_sizes=(50, 100, 50), random_state=99, solver='sgd'), 0.7030623092486541)
              
    Adult:
    PCA Grid Search:  ({'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50),
              random_state=99), 0.8058089823436296)
    ICA Grid Search:  ({'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(hidden_layer_sizes=(50, 100, 50), random_state=99), 0.7932605233070654)
    GRP Grid Search:  ({'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50),
              random_state=99), 0.7986134404161863)
    MBDL Grid Search:  ({'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(hidden_layer_sizes=(50, 100, 50), random_state=99), 0.7966829134392324)
    '''
    kmeans_pca = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    em_pca = GaussianMixture(n_components=3, random_state=random_seed)
    kmeans_ica = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    em_ica = GaussianMixture(n_components=3, random_state=random_seed)
    kmeans_grp = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    em_grp = GaussianMixture(n_components=3, random_state=random_seed)
    kmeans_mbdl = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    em_mbdl = GaussianMixture(n_components=3, random_state=random_seed)

    kmeans_pca.fit(X_1_pca)
    kmeans_ica.fit(X_1_ica)
    kmeans_grp.fit(X_1_grp)
    kmeans_mbdl.fit(X_1_mbdl)
    labels_pca_km = kmeans_pca.labels_
    labels_pca_km = pd.DataFrame(data=labels_pca_km, columns=['Labels'])
    labels_ica_km = kmeans_ica.labels_
    labels_ica_km = pd.DataFrame(data=labels_ica_km, columns=['Labels'])
    labels_grp_km = kmeans_grp.labels_
    labels_grp_km = pd.DataFrame(data=labels_grp_km, columns=['Labels'])
    labels_mbdl_km = kmeans_mbdl.labels_
    labels_mbdl_km = pd.DataFrame(data=labels_mbdl_km, columns=['Labels'])

    X_train_pca_km, X_test_pca_km, y_train_pca_km, y_test_pca_km = trainTestSplit(labels_pca_km, y_1, sample_size, random_seed)
    grid_search_pca_km = gridSearch(grid_param, clf, X_train_pca_km, y_train_pca_km, 'accuracy', dataset_1)

    X_train_ica_km, X_test_ica_km, y_train_ica_km, y_test_ica_km = trainTestSplit(labels_ica_km, y_1, sample_size, random_seed)
    grid_search_ica_km = gridSearch(grid_param, clf, X_train_ica_km, y_train_ica_km, 'accuracy', dataset_1)

    X_train_grp_km, X_test_grp_km, y_train_grp_km, y_test_grp_km = trainTestSplit(labels_grp_km, y_1, sample_size, random_seed)
    grid_search_grp_km = gridSearch(grid_param, clf, X_train_grp_km, y_train_grp_km, 'accuracy', dataset_1)

    X_train_mbdl_km, X_test_mbdl_km, y_train_mbdl_km, y_test_mbdl_km = trainTestSplit(labels_mbdl_km, y_1, sample_size, random_seed)
    grid_search_mbdl_km = gridSearch(grid_param, clf, X_train_mbdl_km, y_train_mbdl_km, 'accuracy', dataset_1)

    print('PCA KMeans Grid Search: ', grid_search_pca_km)
    print('ICA KMeans Grid Search: ', grid_search_ica_km)
    print('GRP KMeans Grid Search: ', grid_search_grp_km)
    print('MBDL KMeans Grid Search: ', grid_search_mbdl_km)

    '''
    Car:
    PCA KMeans Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99), 0.7030623092486541)
    ICA KMeans Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
    GRP KMeans Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
    MBDL KMeans Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99), 0.7030623092486541)
    '''

