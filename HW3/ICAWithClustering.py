import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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
    distance = []
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
        distance.append(clf.inertia_)
        sil_score.append(metrics.silhouette_score(X, y_pred, metric='euclidean'))
        ami_score.append(metrics.adjusted_mutual_info_score(y, y_pred))
        hgy_score.append(metrics.homogeneity_score(y, y_pred))
        comp_score.append(metrics.completeness_score(y, y_pred))
        v_score.append(metrics.v_measure_score(y, y_pred))
    title, xlabel, ylabel = ['ICA KMeans Distance - ' + dataset, 'Number Of Clusters', 'Distance']
    savefile = 'plots/ICA_KMeans_clusters_Distance_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, distance, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['ICA KMeans Silhouette Score - ' + dataset, 'Number Of Clusters', 'Silhouette']
    savefile = 'plots/ICA_KMeans_clusters_silhouette_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, sil_score, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel, label1, label2, label3, label4 = ['ICA KMeans Scores - ' + dataset, 'Number Of Clusters',
                                                                     'Scores', 'Adjusted Mutual Info',
                                                                     'Homogeneity', 'Completeness', 'V Measure']
    savefile = 'plots/ICA_KMeans_clusters_scores_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotCurves(k, ami_score, hgy_score, comp_score, v_score, title, xlabel, ylabel, label1, label2, label3,
               label4, savefile)
    title, xlabel, ylabel = ['ICA KMeans Fit Times - ' + dataset, 'Number Of Clusters', 'Fit Time']
    savefile = 'plots/ICA_KMeans_Fit_Times_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, fit_time, title, xlabel, ylabel, savefile)

def gmm(k, X, y, dataset, random_seed):
    fit_time = []
    pred = []
    aic = []
    bic = []
    sil_score = []
    ami_score = []
    hgy_score = []
    comp_score = []
    v_score = []
    print("Looping through clusters: ", k)
    for i in k:
        print('#k: ', i)
        clf = GaussianMixture(n_components=i, random_state=random_seed)
        fit_start_tm = time.time()
        clf.fit(X)
        fit_end_tm = time.time()
        fit_tm = fit_end_tm - fit_start_tm
        fit_time.append(fit_tm)
        y_pred = clf.predict(X)
        pred.append(y_pred)
        aic.append(clf.aic(X))
        bic.append(clf.bic(X))
        sil_score.append(metrics.silhouette_score(X, y_pred, metric='euclidean'))
        ami_score.append(metrics.adjusted_mutual_info_score(y, y_pred))
        hgy_score.append(metrics.homogeneity_score(y, y_pred))
        comp_score.append(metrics.completeness_score(y, y_pred))
        v_score.append(metrics.v_measure_score(y, y_pred))
    title, xlabel, ylabel, lable1, label2 = ['ICA EM - Gaussian Mixture ' + dataset, 'Number Of Components', 'AIC BIC', 'AIC', 'BIC']
    savefile = 'plots/ICA_EM_Components_AIC_BIC_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotEmCurves(k, aic, bic, title, xlabel, ylabel, lable1, label2, savefile)
    title, xlabel, ylabel = ['ICA EM - Gaussian Mixture Silhouette Score ' + dataset, 'Number Of Clusters', 'Silhouette']
    savefile = 'plots/ICA_EM_Components_Silhouette_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, sil_score, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel, label1, label2, label3, label4 = ['ICA EM - Gaussian Mixture Scores ' + dataset, 'Number Of Components',
                                                                     'Scores', 'Adjusted Mutual Info',
                                                                     'Homogeneity', 'Completeness', 'V Measure']
    savefile = 'plots/ICA_EM_Components_Scores_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotCurves(k, ami_score, hgy_score, comp_score, v_score, title, xlabel, ylabel, label1, label2, label3,
               label4, savefile)
    title, xlabel, ylabel = ['ICA EM - Gaussian Mixture Fit Times - ' + dataset, 'Number Of Components', 'Fit Time']
    savefile = 'plots/ICA_EM_Fit_Times_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, fit_time, title, xlabel, ylabel, savefile)

def plotSilhouette(k, X, dataset, random_seed, cluster):
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
        savefile = 'plots/sil_car/ICA_' + cluster + '_Silhouette_Plot_' + str(k) + '_' + dataset + '.png'
    else:
        savefile = 'plots/sil_adult/ICA_' + cluster + '_Silhouette_Plot_' + str(k) + '_' + dataset + '.png'
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

def plotEmCurves(x, y1, y2, title, xlabel, ylabel, label1, label2, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, 'o-', label=label1)
    plt.plot(x, y2, 'o-', label=label2)
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

def plotBar(labels, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.bar(labels.keys(), labels.values, width=0.2)
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
    ica_1 = FastICA(n_components=7, random_state=random_seed)
    X_1_dr = ica_1.fit_transform(X_1_scaled)
    ica_2 = FastICA(n_components=11, random_state=random_seed)
    X_2_dr = ica_2.fit_transform(X_2_scaled)
    k = np.arange(2, 26)
    kMeans(k, X_1_dr, y_1, dataset_1, random_seed)
    kMeans(k, X_2_dr, y_2, dataset_2, random_seed)

    cluster = 'KMeans'
    plotSilhouette(2, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(3, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(4, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(7, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(8, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(9, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(10, X_1_dr, dataset_1, random_seed, cluster)

    plotSilhouette(2, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(3, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(4, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(5, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(10, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(11, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(12, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(23, X_2_dr, dataset_2, random_seed, cluster)
    '''
        KMeans
        car - cluster = 3
        adult - cluster = 2
    '''
    clf_1 = KMeans(n_clusters=4, init='k-means++', random_state=random_seed)
    clf_2 = KMeans(n_clusters=2, init='k-means++', random_state=random_seed)
    clf_1.fit(X_1_dr)
    clf_2.fit(X_2_dr)
    cluster_labels_1 = clf_1.fit_predict(X_1_dr)
    cluster_labels_1 = pd.DataFrame(data=cluster_labels_1, columns=['Labels'])
    cluster_labels_1_counts = cluster_labels_1['Labels'].value_counts()
    cluster_labels_2 = clf_2.fit_predict(X_2_dr)
    cluster_labels_2 = pd.DataFrame(data=cluster_labels_2, columns=['Labels'])
    cluster_labels_2_counts = cluster_labels_2['Labels'].value_counts()
    title, xlabel, ylabel = ['ICA KMeans - ' + dataset_1, 'Label', 'Count']
    savefile = 'plots/ICA_KMeans_Label_' + dataset_1 + '.png'
    plotBar(cluster_labels_1_counts, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['ICA KMeans - ' + dataset_2, 'Label', 'Count']
    savefile = 'plots/ICA_KMeans_Label_' + dataset_2 + '.png'
    plotBar(cluster_labels_2_counts, title, xlabel, ylabel, savefile)

    gmm(k, X_1_dr, y_1, dataset_1, random_seed)
    gmm(k, X_2_dr, y_2, dataset_2, random_seed)

    cluster = 'EM'
    plotSilhouette(2, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(3, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(4, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(7, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(8, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(9, X_1_dr, dataset_1, random_seed, cluster)
    plotSilhouette(10, X_1_dr, dataset_1, random_seed, cluster)

    plotSilhouette(2, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(3, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(4, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(5, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(10, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(11, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(12, X_2_dr, dataset_2, random_seed, cluster)
    plotSilhouette(23, X_2_dr, dataset_2, random_seed, cluster)
    '''
        EM
        car - cluster = 4
        adult - cluster = 2
    '''
    clf_1 = GaussianMixture(n_components=4, random_state=random_seed)
    clf_2 = GaussianMixture(n_components=2, random_state=random_seed)
    clf_1.fit(X_1_dr)
    clf_2.fit(X_2_dr)
    cluster_labels_1 = clf_1.fit_predict(X_1_dr)
    cluster_labels_1 = pd.DataFrame(data=cluster_labels_1, columns=['Labels'])
    cluster_labels_1_counts = cluster_labels_1['Labels'].value_counts()
    cluster_labels_2 = clf_2.fit_predict(X_2_dr)
    cluster_labels_2 = pd.DataFrame(data=cluster_labels_2, columns=['Labels'])
    cluster_labels_2_counts = cluster_labels_2['Labels'].value_counts()
    title, xlabel, ylabel = ['ICA EM - ' + dataset_1, 'Label', 'Count']
    savefile = 'plots/ICA_EM_Label_' + dataset_1 + '.png'
    plotBar(cluster_labels_1_counts, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['ICA EM - ' + dataset_2, 'Label', 'Count']
    savefile = 'plots/ICA_EM_Label_' + dataset_2 + '.png'
    plotBar(cluster_labels_2_counts, title, xlabel, ylabel, savefile)