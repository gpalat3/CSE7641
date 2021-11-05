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
    learn_start_tm = time.time()
    clf.fit(X_train, y_train)
    learn_end_tm = time.time()
    learn_tm = learn_end_tm - learn_start_tm
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
    return y_pred, y_pred_train, learn_tm, query_tm, query_tm_train, rmse, rmse_train

if __name__ == '__main__':
    col_names_car = ['buying_cost', 'maintenance_cost', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    X_1, y_1 = loadCarData('data/car_evaluation/car_data.csv', col_names_car)
    col_names_adult = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'label']
    X_2, y_2 = loadAdultData('data/adult/adult_data.csv', col_names_adult)
    random_seed = 99
    sample_size = 0.3
    dataset_1 = 'car'
    dataset_2 = 'adult'
    scale = MinMaxScaler()
    X_1_scaled = scale.fit_transform(X_1)
    X_2_scaled = scale.fit_transform(X_2)
    X_train_1, X_test_1, y_train_1, y_test_1 = trainTestSplit(X_1_scaled, y_1, sample_size, random_seed)
    X_train_2, X_test_2, y_train_2, y_test_2 = trainTestSplit(X_2_scaled, y_2, sample_size, random_seed)
    pca_1 = PCA(n_components=9, random_state=random_seed)
    pca_2 = PCA(n_components=30, random_state=random_seed)
    ica_1 = FastICA(n_components=7, random_state=random_seed)
    ica_2 = FastICA(n_components=10, random_state=random_seed)
    grp_1 = GaussianRandomProjection(n_components=9, random_state=random_seed)
    grp_2 = GaussianRandomProjection(n_components=9, random_state=random_seed)
    mbdl_1 = MiniBatchDictionaryLearning(n_components=9, random_state=random_seed, batch_size=200, n_iter=100, alpha=1)
    mbdl_2 = MiniBatchDictionaryLearning(n_components=4, random_state=random_seed, batch_size=200, n_iter=100, alpha=1)
    X_train_1_pca = pca_1.fit_transform(X_train_1)
    X_test_1_pca = pca_1.fit_transform(X_test_1)
    X_train_2_pca = pca_2.fit_transform(X_train_2)
    X_test_2_pca = pca_2.fit_transform(X_test_2)

    X_train_1_ica = ica_1.fit_transform(X_train_1)
    X_test_1_ica = ica_1.fit_transform(X_test_1)
    X_train_2_ica = ica_2.fit_transform(X_train_2)
    X_test_2_ica = ica_2.fit_transform(X_test_2)

    X_train_1_grp = grp_1.fit_transform(X_train_1)
    X_test_1_grp = grp_1.fit_transform(X_test_1)
    X_train_2_grp = grp_2.fit_transform(X_train_2)
    X_test_2_grp = grp_2.fit_transform(X_test_2)

    X_train_1_mbdl = mbdl_1.fit_transform(X_train_1)
    X_test_1_mbdl = mbdl_1.fit_transform(X_test_1)
    X_train_2_mbdl = mbdl_2.fit_transform(X_train_2)
    X_test_2_mbdl = mbdl_2.fit_transform(X_test_2)

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
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    em = GaussianMixture(n_components=3, random_state=random_seed)
    kmeans_pca = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    kmeans_ica = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    kmeans_grp = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    kmeans_mbdl = KMeans(n_clusters=7, init='k-means++', random_state=random_seed)
    em_pca = GaussianMixture(n_components=3, random_state=random_seed)
    em_ica = GaussianMixture(n_components=3, random_state=random_seed)
    em_grp = GaussianMixture(n_components=3, random_state=random_seed)
    em_mbdl = GaussianMixture(n_components=7, random_state=random_seed)

    kmeans.fit(X_train_1)
    kmeans_pca.fit(X_train_1_pca)
    kmeans_ica.fit(X_train_1_ica)
    kmeans_grp.fit(X_train_1_grp)
    kmeans_mbdl.fit(X_train_1_mbdl)

    X_train_1_km = kmeans.transform(X_train_1)
    X_train_1_pca_km = kmeans_pca.transform(X_train_1_pca)
    X_train_1_ica_km = kmeans_ica.transform(X_train_1_ica)
    X_train_1_grp_km = kmeans_grp.transform(X_train_1_grp)
    X_train_1_mbdl_km = kmeans_mbdl.transform(X_train_1_mbdl)

    X_test_1_km = kmeans.transform(X_test_1)
    X_test_1_pca_km = kmeans_pca.transform(X_test_1_pca)
    X_test_1_ica_km = kmeans_ica.transform(X_test_1_ica)
    X_test_1_grp_km = kmeans_grp.transform(X_test_1_grp)
    X_test_1_mbdl_km = kmeans_mbdl.transform(X_test_1_mbdl)

    em.fit(X_train_1)
    em_pca.fit(X_train_1_pca)
    em_ica.fit(X_train_1_ica)
    em_grp.fit(X_train_1_grp)
    em_mbdl.fit(X_train_1_mbdl)

    X_train_1_em = em.predict_proba(X_train_1)
    X_train_1_pca_em = em_pca.predict_proba(X_train_1_pca)
    X_train_1_ica_em = em_ica.predict_proba(X_train_1_ica)
    X_train_1_grp_em = em_grp.predict_proba(X_train_1_grp)
    X_train_1_mbdl_em = em_mbdl.predict_proba(X_train_1_mbdl)

    X_test_1_em = em.predict_proba(X_test_1)
    X_test_1_pca_em = em_pca.predict_proba(X_test_1_pca)
    X_test_1_ica_em = em_ica.predict_proba(X_test_1_ica)
    X_test_1_grp_em = em_grp.predict_proba(X_test_1_grp)
    X_test_1_mbdl_em = em_mbdl.predict_proba(X_test_1_mbdl)

    grid_search_km = gridSearch(grid_param, clf, X_train_1_km, y_train_1, 'accuracy', dataset_1)
    grid_search_em = gridSearch(grid_param, clf, X_train_1_em, y_train_1, 'accuracy', dataset_1)
    grid_search_pca = gridSearch(grid_param, clf, X_train_1_pca, y_train_1, 'accuracy', dataset_1)
    grid_search_ica = gridSearch(grid_param, clf, X_train_1_ica, y_train_1, 'accuracy', dataset_1)
    grid_search_grp = gridSearch(grid_param, clf, X_train_1_grp, y_train_1, 'accuracy', dataset_1)
    grid_search_mbdl = gridSearch(grid_param, clf, X_train_1_mbdl, y_train_1, 'accuracy', dataset_1)
    grid_search_pca_em = gridSearch(grid_param, clf, X_train_1_pca_em, y_train_1, 'accuracy', dataset_1)
    grid_search_ica_em = gridSearch(grid_param, clf, X_train_1_ica_em, y_train_1, 'accuracy', dataset_1)
    grid_search_grp_em = gridSearch(grid_param, clf, X_train_1_grp_em, y_train_1, 'accuracy', dataset_1)
    grid_search_mbdl_em = gridSearch(grid_param, clf, X_train_1_mbdl_em, y_train_1, 'accuracy', dataset_1)
    grid_search_pca_km = gridSearch(grid_param, clf, X_train_1_pca_km, y_train_1, 'accuracy', dataset_1)
    grid_search_ica_km = gridSearch(grid_param, clf, X_train_1_ica_km, y_train_1, 'accuracy', dataset_1)
    grid_search_grp_km = gridSearch(grid_param, clf, X_train_1_grp_km, y_train_1, 'accuracy', dataset_1)
    grid_search_mbdl_km = gridSearch(grid_param, clf, X_train_1_mbdl_km, y_train_1, 'accuracy', dataset_1)

    print('----------------------------------------')
    print('KMeans Grid Search: ', grid_search_km)
    print('EM Grid Search: ', grid_search_em)
    print('PCA Grid Search: ', grid_search_pca)
    print('ICA Grid Search: ', grid_search_ica)
    print('GRP Grid Search: ', grid_search_grp)
    print('MBDL Grid Search: ', grid_search_mbdl)
    print('----------------------------------------')
    print('PCA KMeans Grid Search: ', grid_search_pca_km)
    print('ICA KMeans Grid Search: ', grid_search_ica_km)
    print('GRP KMeans Grid Search: ', grid_search_grp_km)
    print('MBDL KMeans Grid Search: ', grid_search_mbdl_km)
    print('----------------------------------------')
    print('PCA EM Grid Search: ', grid_search_pca_km)
    print('ICA EM Grid Search: ', grid_search_ica_km)
    print('GRP EM Grid Search: ', grid_search_grp_km)
    print('MBDL EM Grid Search: ', grid_search_mbdl_km)
    print('----------------------------------------')

    '''
        best_clf_pca = clf
        best_clf_ica = clf
        best_clf_grp = clf
        best_clf_mbdl = clf
        best_clf_pca_km = clf
        best_clf_ica_km = clf
        best_clf_grp_km = clf
        best_clf_mbdl_km = clf
        best_clf_pca_em = clf
        best_clf_ica_em = clf
        best_clf_grp_em = clf
        best_clf_mbdl_em = clf
    '''
    best_clf_km = grid_search_km[1]
    best_clf_em = grid_search_em[1]
    best_clf_pca = grid_search_pca[1]
    best_clf_ica = grid_search_ica[1]
    best_clf_grp = grid_search_grp[1]
    best_clf_mbdl = grid_search_mbdl[1]
    best_clf_pca_km = grid_search_pca_km[1]
    best_clf_ica_km = grid_search_ica_km[1]
    best_clf_grp_km = grid_search_grp_km[1]
    best_clf_mbdl_km = grid_search_mbdl_km[1]
    best_clf_pca_em = grid_search_pca_em[1]
    best_clf_ica_em = grid_search_ica_em[1]
    best_clf_grp_em = grid_search_grp_em[1]
    best_clf_mbdl_em = grid_search_mbdl_em[1]

    y_pred_km, y_pred_train_km, learn_tm_km, query_tm_km, query_tm_train_km, rmse_km, rmse_train_km = mlModel(
        best_clf_km, X_train_1_km, X_test_1_km, y_train_1, y_test_1)
    y_pred_em, y_pred_train_em, learn_tm_em, query_tm_em, query_tm_train_em, rmse_em, rmse_train_em = mlModel(
        best_clf_em, X_train_1_em, X_test_1_em, y_train_1, y_test_1)
    y_pred_pca, y_pred_train_pca, learn_tm_pca, query_tm_pca, query_tm_train_pca, rmse_pca, rmse_train_pca = mlModel(
        best_clf_pca, X_train_1_pca, X_test_1_pca, y_train_1, y_test_1)
    y_pred_ica, y_pred_train_ica, learn_tm_ica, query_tm_ica, query_tm_train_ica, rmse_ica, rmse_train_ica = mlModel(
        best_clf_ica, X_train_1_ica, X_test_1_ica, y_train_1, y_test_1)
    y_pred_grp, y_pred_train_grp, learn_tm_grp, query_tm_grp, query_tm_train_grp, rmse_grp, rmse_train_grp = mlModel(
        best_clf_grp, X_train_1_grp, X_test_1_grp, y_train_1, y_test_1)
    y_pred_mbdl, y_pred_train_mbdl, learn_tm_mbdl, query_tm_mbdl, query_tm_train_mbdl, rmse_mbdl, rmse_train_mbdl = mlModel(
        best_clf_mbdl, X_train_1_mbdl, X_test_1_mbdl, y_train_1, y_test_1)

    y_pred_pca_km, y_pred_train_pca_km, learn_tm_pca_km, query_tm_pca_km, query_tm_train_pca_km, rmse_pca_km, rmse_train_pca_km = mlModel(
        best_clf_pca_km, X_train_1_pca_km, X_test_1_pca_km, y_train_1, y_test_1)
    y_pred_ica_km, y_pred_train_ica_km, learn_tm_ica_km, query_tm_ica_km, query_tm_train_ica_km, rmse_ica_km, rmse_train_ica_km = mlModel(
        best_clf_ica_km, X_train_1_ica_km, X_test_1_ica_km, y_train_1, y_test_1)
    y_pred_grp_km, y_pred_train_grp_km, learn_tm_grp_km, query_tm_grp_km, query_tm_train_grp_km, rmse_grp_km, rmse_train_grp_km = mlModel(
        best_clf_grp_km, X_train_1_grp_km, X_test_1_grp_km, y_train_1, y_test_1)
    y_pred_mbdl_km, y_pred_train_mbdl_km, learn_tm_mbdl_km, query_tm_mbdl_km, query_tm_train_mbdl_km, rmse_mbdl_km, rmse_train_mbdl_km = mlModel(
        best_clf_mbdl_km, X_train_1_mbdl_km, X_test_1_mbdl_km, y_train_1, y_test_1)

    y_pred_pca_em, y_pred_train_pca_em, learn_tm_pca_em, query_tm_pca_em, query_tm_train_pca_em, rmse_pca_em, rmse_train_pca_em = mlModel(
        best_clf_pca_em, X_train_1_pca_em, X_test_1_pca_em, y_train_1, y_test_1)
    y_pred_ica_em, y_pred_train_ica_em, learn_tm_ica_em, query_tm_ica_em, query_tm_train_ica_em, rmse_ica_em, rmse_train_ica_em = mlModel(
        best_clf_ica_em, X_train_1_ica_em, X_test_1_ica_em, y_train_1, y_test_1)
    y_pred_grp_em, y_pred_train_grp_em, learn_tm_grp_em, query_tm_grp_em, query_tm_train_grp_em, rmse_grp_em, rmse_train_grp_em = mlModel(
        best_clf_grp_em, X_train_1_grp_em, X_test_1_grp_em, y_train_1, y_test_1)
    y_pred_mbdl_em, y_pred_train_mbdl_em, learn_tm_mbdl_em, query_tm_mbdl_em, query_tm_train_mbdl_em, rmse_mbdl_em, rmse_train_mbdl_em = mlModel(
        best_clf_mbdl_em, X_train_1_mbdl_em, X_test_1_mbdl_em, y_train_1, y_test_1)

    print('----------------------------------------')
    print("MLP KMeans Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_km))
    print("MLP KMeans Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_km))
    print("MLP KMeans Learning Time Dataset 1:", learn_tm_km)
    print("MLP KMeans Query Time Dataset 1:", query_tm_km)
    print('----------------------------------------')
    print('----------------------------------------')
    print("MLP EM Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_em))
    print("MLP EM Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_em))
    print("MLP EM Learning Time Dataset 1:", learn_tm_em)
    print("MLP EM Query Time Dataset 1:", query_tm_em)
    print('----------------------------------------')
    print('----------------------------------------')
    print("MLP PCA Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_pca))
    print("MLP PCA Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_pca))
    print("MLP PCA Learning Time Dataset 1:", learn_tm_pca)
    print("MLP PCA Query Time Dataset 1:", query_tm_pca)
    print('----------------------------------------')
    print("MLP ICA Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_ica))
    print("MLP ICA Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_ica))
    print("MLP ICA Learning Time Dataset 1:", learn_tm_ica)
    print("MLP ICA Query Time Dataset 1:", query_tm_ica)
    print('----------------------------------------')
    print("MLP GRP Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_grp))
    print("MLP GRP Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_grp))
    print("MLP GRP Learning Time Dataset 1:", learn_tm_grp)
    print("MLP GRP Query Time Dataset 1:", query_tm_grp)
    print('----------------------------------------')
    print("MLP MBDL Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_mbdl))
    print("MLP MBDL Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_mbdl))
    print("MLP MBDL Learning Time Dataset 1:", learn_tm_mbdl)
    print("MLP MBDL Query Time Dataset 1:", query_tm_mbdl)
    print('----------------------------------------')
    print('----------------------------------------')
    print("MLP PCA KMeans Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_pca_km))
    print("MLP PCA KMeans Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_pca_km))
    print("MLP PCA KMeans Learning Time Dataset 1:", learn_tm_pca_km)
    print("MLP PCA KMeans Query Time Dataset 1:", query_tm_pca_km)
    print('----------------------------------------')
    print("MLP ICA KMeans Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_ica_km))
    print("MLP ICA KMeans Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_ica_km))
    print("MLP ICA KMeans Learning Time Dataset 1:", learn_tm_ica_km)
    print("MLP ICA KMeans Query Time Dataset 1:", query_tm_ica_km)
    print('----------------------------------------')
    print("MLP GRP KMeans Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_grp_km))
    print("MLP GRP KMeans Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_grp_km))
    print("MLP GRP KMeans Learning Time Dataset 1:", learn_tm_grp_km)
    print("MLP GRP KMeans Query Time Dataset 1:", query_tm_grp_km)
    print('----------------------------------------')
    print("MLP MBDL KMeans Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_mbdl_km))
    print("MLP MBDL KMeans Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_mbdl_km))
    print("MLP MBDL KMeans Learning Time Dataset 1:", learn_tm_mbdl_km)
    print("MLP MBDL KMeans Query Time Dataset 1:", query_tm_mbdl_km)
    print('----------------------------------------')
    print('----------------------------------------')
    print("MLP PCA EM Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_pca_em))
    print("MLP PCA EM Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_pca_em))
    print("MLP PCA EM Learning Time Dataset 1:", learn_tm_pca_em)
    print("MLP PCA EM Query Time Dataset 1:", query_tm_pca_em)
    print('----------------------------------------')
    print("MLP ICA EM Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_ica_em))
    print("MLP ICA EM Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_ica_em))
    print("MLP ICA EM Learning Time Dataset 1:", learn_tm_ica_em)
    print("MLP ICA EM Query Time Dataset 1:", query_tm_ica_em)
    print('----------------------------------------')
    print("MLP GRP EM Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_grp_em))
    print("MLP GRP EM Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_grp_em))
    print("MLP GRP EM Learning Time Dataset 1:", learn_tm_grp_em)
    print("MLP GRP EM Query Time Dataset 1:", query_tm_grp_em)
    print('----------------------------------------')
    print("MLP MBDL EM Accuracy Dataset 1: In Sample", metrics.accuracy_score(y_train_1, y_pred_train_mbdl_em))
    print("MLP MBDL EM Accuracy Dataset 1: Out Sample", metrics.accuracy_score(y_test_1, y_pred_mbdl_em))
    print("MLP MBDL EM Learning Time Dataset 1:", learn_tm_mbdl_em)
    print("MLP MBDL EM Query Time Dataset 1:", query_tm_mbdl_em)
    print('----------------------------------------')

    '''
    Car:
----------------------------------------
KMeans Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 100, 50),
              random_state=99), 0.7038887555296458)
EM Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
PCA Grid Search:  ({'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50),
              random_state=99), 0.7502314735434312)
ICA Grid Search:  ({'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(alpha=0.01, hidden_layer_sizes=(50, 100, 50), random_state=99), 0.7088577209286375)
GRP Grid Search:  ({'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 100, 50),
              random_state=99), 0.7626315969959878)
MBDL Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 100, 50),
              random_state=99), 0.7030623092486541)
----------------------------------------
PCA KMeans Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
ICA KMeans Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
GRP KMeans Grid Search:  ({'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(alpha=0.05, hidden_layer_sizes=(50, 100, 50), random_state=99), 0.7063715236103014)
MBDL KMeans Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
----------------------------------------
PCA EM Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
ICA EM Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
GRP EM Grid Search:  ({'activation': 'relu', 'alpha': 0.05, 'hidden_layer_sizes': (50, 100, 50), 'solver': 'adam'}, MLPClassifier(alpha=0.05, hidden_layer_sizes=(50, 100, 50), random_state=99), 0.7063715236103014)
MBDL EM Grid Search:  ({'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'solver': 'sgd'}, MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50),
              random_state=99, solver='sgd'), 0.7030623092486541)
----------------------------------------

----------------------------------------
MLP KMeans Accuracy Dataset 1: In Sample 0.7030603804797353
MLP KMeans Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP KMeans Learning Time Dataset 1: 0.6267304420471191
MLP KMeans Query Time Dataset 1: 0.0019996166229248047
----------------------------------------
----------------------------------------
MLP EM Accuracy Dataset 1: In Sample 0.7030603804797353
MLP EM Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP EM Learning Time Dataset 1: 2.853698968887329
MLP EM Query Time Dataset 1: 0.0020074844360351562
----------------------------------------
----------------------------------------
MLP PCA Accuracy Dataset 1: In Sample 0.7890818858560794
MLP PCA Accuracy Dataset 1: Out Sample 0.653179190751445
MLP PCA Learning Time Dataset 1: 3.0294907093048096
MLP PCA Query Time Dataset 1: 0.0020189285278320312
----------------------------------------
MLP ICA Accuracy Dataset 1: In Sample 0.7105045492142267
MLP ICA Accuracy Dataset 1: Out Sample 0.5703275529865125
MLP ICA Learning Time Dataset 1: 3.6865408420562744
MLP ICA Query Time Dataset 1: 0.0019974708557128906
----------------------------------------
MLP GRP Accuracy Dataset 1: In Sample 0.7791563275434243
MLP GRP Accuracy Dataset 1: Out Sample 0.7552986512524085
MLP GRP Learning Time Dataset 1: 4.129885673522949
MLP GRP Query Time Dataset 1: 0.003034830093383789
----------------------------------------
MLP MBDL Accuracy Dataset 1: In Sample 0.6699751861042184
MLP MBDL Accuracy Dataset 1: Out Sample 0.5240847784200385
MLP MBDL Learning Time Dataset 1: 0.7987163066864014
MLP MBDL Query Time Dataset 1: 0.0019693374633789062
----------------------------------------
----------------------------------------
MLP PCA KMeans Accuracy Dataset 1: In Sample 0.7030603804797353
MLP PCA KMeans Accuracy Dataset 1: Out Sample 0.6840077071290944
MLP PCA KMeans Learning Time Dataset 1: 3.4405906200408936
MLP PCA KMeans Query Time Dataset 1: 0.0019986629486083984
----------------------------------------
MLP ICA KMeans Accuracy Dataset 1: In Sample 0.7030603804797353
MLP ICA KMeans Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP ICA KMeans Learning Time Dataset 1: 0.8563556671142578
MLP ICA KMeans Query Time Dataset 1: 0.0020003318786621094
----------------------------------------
MLP GRP KMeans Accuracy Dataset 1: In Sample 0.7030603804797353
MLP GRP KMeans Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP GRP KMeans Learning Time Dataset 1: 2.2791407108306885
MLP GRP KMeans Query Time Dataset 1: 0.0010008811950683594
----------------------------------------
MLP MBDL KMeans Accuracy Dataset 1: In Sample 0.7030603804797353
MLP MBDL KMeans Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP MBDL KMeans Learning Time Dataset 1: 3.3492400646209717
MLP MBDL KMeans Query Time Dataset 1: 0.0019989013671875
----------------------------------------
----------------------------------------
MLP PCA EM Accuracy Dataset 1: In Sample 0.7030603804797353
MLP PCA EM Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP PCA EM Learning Time Dataset 1: 3.18721866607666
MLP PCA EM Query Time Dataset 1: 0.001999378204345703
----------------------------------------
MLP ICA EM Accuracy Dataset 1: In Sample 0.7030603804797353
MLP ICA EM Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP ICA EM Learning Time Dataset 1: 1.8210444450378418
MLP ICA EM Query Time Dataset 1: 0.0019991397857666016
----------------------------------------
MLP GRP EM Accuracy Dataset 1: In Sample 0.7030603804797353
MLP GRP EM Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP GRP EM Learning Time Dataset 1: 0.819000244140625
MLP GRP EM Query Time Dataset 1: 0.002002716064453125
----------------------------------------
MLP MBDL EM Accuracy Dataset 1: In Sample 0.7030603804797353
MLP MBDL EM Accuracy Dataset 1: Out Sample 0.6936416184971098
MLP MBDL EM Learning Time Dataset 1: 3.398866653442383
MLP MBDL EM Query Time Dataset 1: 0.0020313262939453125
----------------------------------------

Process finished with exit code 0

    '''

