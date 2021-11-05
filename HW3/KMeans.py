import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
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
    distortion = []
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
        clf.fit(X)
        fit_end_tm = time.time()
        fit_tm = fit_end_tm - fit_start_tm
        fit_time.append(fit_tm)
        y_pred = clf.fit_predict(X)
        distortion.append(clf.inertia_)
        sil_score.append(metrics.silhouette_score(X, y_pred, metric='euclidean'))
        ami_score.append(metrics.adjusted_mutual_info_score(y, y_pred))
        hgy_score.append(metrics.homogeneity_score(y, y_pred))
        comp_score.append(metrics.completeness_score(y, y_pred))
        v_score.append(metrics.v_measure_score(y, y_pred))
    title, xlabel, ylabel = ['KMeans Distortion - ' + dataset, 'Number Of Clusters', 'Distance']
    savefile = 'plots/KMeans_Clusters_Distortion_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, distortion, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['KMeans Silhouette Score - ' + dataset, 'Number Of Clusters', 'Silhouette']
    savefile = 'plots/KMeans_clusters_silhouette_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, sil_score, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel, label1, label2, label3, label4 = ['KMeans Scores - ' + dataset, 'Number Of Clusters',
                                                                     'Scores', 'Adjusted Mutual Info',
                                                                     'Homogeneity', 'Completeness', 'V Measure']
    savefile = 'plots/KMeans_clusters_scores_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotCurves(k, ami_score, hgy_score, comp_score, v_score, title, xlabel, ylabel, label1, label2, label3,
               label4, savefile)
    title, xlabel, ylabel = ['KMeans Fit Times - ' + dataset, 'Number Of Clusters', 'Fit Time']
    savefile = 'plots/KMeans_Fit_Times_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, fit_time, title, xlabel, ylabel, savefile)

def plotSilhouette(k, X, dataset, random_seed):
    print('Plotting k=%s for dataset %s ' % (k, dataset))
    fig, ax1 = plt.subplots()
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(X) + (k + 1) * 10])
    clf = KMeans(n_clusters=k, init='k-means++', random_state=random_seed)
    cluster_labels = clf.fit_predict(X)
    centers = clf.cluster_centers_
    sil_vals = metrics.silhouette_samples(X, cluster_labels, metric='euclidean')
    y_lower = 10
    for i in range(k):
        i_sil_vals = sil_vals[cluster_labels == i]
        i_sil_vals.sort()
        i_cluster_size = i_sil_vals.shape[0]
        y_upper = y_lower + i_cluster_size
        color = cm.jet(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, i_sil_vals, facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * i_cluster_size, str(i))
        y_lower = y_upper + 10
    sil_avg = np.mean(sil_vals)
    ax1.axvline(sil_avg, color="red", linestyle="--")
    ax1.set_title('KMeans Silhouette Plot ' + dataset)
    ax1.set_ylabel('Cluster')
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_yticks([])
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    if dataset == 'car':
        savefile = 'plots/sil_car/KMeans_Silhouette_Plot_' + str(k) + '_' + dataset + '.png'
    else:
        savefile = 'plots/sil_adult/KMeans_Silhouette_Plot_' + str(k) + '_' + dataset + '.png'
    plt.savefig(savefile)

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
    plt.grid()
    plt.savefig(savefile)

def plotIndCurves(x, y, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, 'o-')
    plt.grid()
    plt.savefig(savefile)

def plotScatter(X, y, cluster_size, cluster_centers, title, xlabel, ylabel, savefile):
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    for i in cluster_size:
        ax.scatter(X[y == i, 0], X[y == i, 1], marker='o', edgecolor='black', label='cluster ' + str(i))
    # sns.scatterplot(X[:,0], X[:,-1])
    # sns.scatterplot(cluster_centers[:,0], cluster_centers[:,1], color='red', edgecolor='black', marker='*', label='cluster centers')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', edgecolor='black', marker='*', label='centroids')
    ax.legend(loc='best')
    '''
    colors = sns.color_palette('husl', n_colors=len(cluster_size))
    cmap = dict(zip(cluster_size, colors))

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    '''
    for i in cluster_size:
        plt.scatter(X[y == i, 0], X[y == i, 1], s=50, marker='o', edgecolor='black', label='cluster ' + str(i))
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=250, color='yellow', edgecolor='black', marker='*',
               label='centroids')
    '''
    plt.legend(loc='best')
    plt.grid()
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
    k = np.arange(2, 25)
    '''
    kMeans(k, X_1_scaled, y_1, dataset_1, random_seed)
    plotSilhouette(2, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(3, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(4, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(8, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(9, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(10, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(11, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(12, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(13, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(14, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(15, X_1_scaled, dataset_1, random_seed)
    kMeans(k, X_2_scaled, y_2, dataset_2, random_seed)
    plotSilhouette(2, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(3, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(4, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(5, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(10, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(19, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(23, X_2_scaled, dataset_2, random_seed)
    '''
    '''
    car - cluster = 3
    2, 8, 10
    adult - cluster = 2
    2, 3, 5 
    '''
    clf_1 = KMeans(n_clusters=3, init='k-means++', random_state=random_seed)
    clf_2 = KMeans(n_clusters=2, init='k-means++', random_state=random_seed)
    clf_1.fit(X_1_scaled)
    clf_2.fit(X_2_scaled)
    cluster_labels_1 = clf_1.fit_predict(X_1_scaled)
    cluster_centers_1 = clf_1.cluster_centers_
    cluster_size_1 = np.unique(cluster_labels_1)
    cluster_labels_2 = clf_2.fit_predict(X_2_scaled)
    cluster_centers_2 = clf_2.cluster_centers_
    cluster_size_2 = np.unique(cluster_labels_2)
    title, xlabel, ylabel = ['KMeans Scatter Plot ' + dataset_1, 'KM1', 'KM2']
    savefile = 'plots/KMeans_Scatter_Plot_' + str(cluster_size_1) + '_' + dataset_1 + '.png'
    plotScatter(X_1_scaled, cluster_labels_1, cluster_size_1, cluster_centers_1, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['KMeans Scatter Plot ' + dataset_2, 'KM1', 'KM2']
    savefile = 'plots/KMeans_Scatter_Plot_' + str(cluster_size_2) + '_' + dataset_2 + '.png'
    plotScatter(X_2_scaled, y_2, cluster_size_2, cluster_centers_2, title, xlabel, ylabel, savefile)
