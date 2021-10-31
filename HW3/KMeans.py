import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
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

def kMeans(k, X, y, dataset, random_seed):
    fit_time = []
    pred = []
    inertia = []
    sil_score = []
    ami_score = []
    hgy_score = []
    comp_score = []
    v_score = []
    print("Looping through clusters: ", k)
    for i in k:
        print('#k: ', i)
        clf = KMeans(n_clusters=i, init='k-means++', random_state=random_seed)
        fit_start_tm = time.time()
        clf.fit(X, y)
        fit_end_tm = time.time()
        fit_tm = fit_end_tm - fit_start_tm
        fit_time.append(fit_tm)
        y_pred = clf.labels_
        pred.append(y_pred)
        inertia.append(clf.inertia_)
        sil_score.append(metrics.silhouette_score(X, y_pred, metric='euclidean'))
        ami_score.append(metrics.adjusted_mutual_info_score(y, y_pred))
        hgy_score.append(metrics.homogeneity_score(y, y_pred))
        comp_score.append(metrics.completeness_score(y, y_pred))
        v_score.append(metrics.v_measure_score(y, y_pred))
    title, xlabel, ylabel = ['KMeans Loss - ' + dataset, 'Number Of Clusters', 'Sum Of Squared Error']
    savefile = 'plots/KMeans_clusters_loss_' + dataset + '.png'
    print("Plotting % for dataset % ", title, dataset)
    plotIndCurves(k, inertia, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['KMeans Silhouette Score - ' + dataset, 'Number Of Clusters', 'Silhouette']
    savefile = 'plots/KMeans_clusters_silhouette_' + dataset + '.png'
    print("Plotting % for dataset % ", title, dataset)
    plotIndCurves(k, sil_score, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel, label1, label2, label3, label4 = ['KMeans Scores - ' + dataset, 'Number Of Clusters',
                                                                     'Scores', 'Adjusted Mutual Info',
                                                                     'Homogeneity', 'Completeness', 'V Measure']
    savefile = 'plots/KMeans_clusters_scores_' + dataset + '.png'
    print("Plotting % for dataset % ", title, dataset)
    plotCurves(k, ami_score, hgy_score, comp_score, v_score, title, xlabel, ylabel, label1, label2, label3,
               label4, savefile)
    title, xlabel, ylabel = ['KMeans Fit Times - ' + dataset, 'Number Of Clusters', 'Fit Time']
    savefile = 'plots/KMeans_fit_times_' + dataset + '.png'
    print("Plotting % for dataset % ", title, dataset)
    plotIndCurves(k, fit_time, title, xlabel, ylabel, savefile)

def plotCurves(x1, y1, y2, y3, y4, title, xlabel, ylabel, label1, label2, label3, label4, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x1, y1, 'o-', label=label1)
    plt.plot(x1, y2, 'o-', label=label2)
    plt.plot(x1, y3, 'o-', label=label3)
    plt.plot(x1, y4, 'o-', label=label4)
    plt.legend(loc='best')
    plt.savefig(savefile)

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
    scale = StandardScaler()
    X_1_scaled = scale.fit_transform(X_1)
    X_2_scaled = scale.fit_transform(X_2)
    '''
    X_train_1_scaled = scale.fit_transform(X_train_1)
    X_test_1_scaled = scale.fit_transform(X_test_1)
    X_train_2_scaled = scale.fit_transform(X_train_2)
    X_test_2_scaled = scale.fit_transform(X_test_2)
    '''
    k = np.arange(2, 51)
    kMeans(k, X_1_scaled, y_1, dataset_1, random_seed)
    kMeans(k, X_2_scaled, y_2, dataset_2, random_seed)
