import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
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
    title, xlabel, ylabel, lable1, label2 = ['EM - Gaussian Mixture ' + dataset, 'Number Of Components', 'AIC BIC', 'AIC', 'BIC']
    savefile = 'plots/EM_Components_AIC_BIC_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotEmCurves(k, aic, bic, title, xlabel, ylabel, lable1, label2, savefile)
    title, xlabel, ylabel = ['EM - Gaussian Mixture Silhouette Score ' + dataset, 'Number Of Clusters', 'Silhouette']
    savefile = 'plots/EM_Components_Silhouette_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, sil_score, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel, label1, label2, label3, label4 = ['EM - Gaussian Mixture Scores ' + dataset, 'Number Of Components',
                                                                     'Scores', 'Adjusted Mutual Info',
                                                                     'Homogeneity', 'Completeness', 'V Measure']
    savefile = 'plots/EM_Components_Scores_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotCurves(k, ami_score, hgy_score, comp_score, v_score, title, xlabel, ylabel, label1, label2, label3,
               label4, savefile)
    title, xlabel, ylabel = ['EM - Gaussian Mixture Fit Times - ' + dataset, 'Number Of Components', 'Fit Time']
    savefile = 'plots/EM_Fit_Times_' + dataset + '.png'
    print("Plotting %s for dataset %s " % (title, dataset))
    plotIndCurves(k, fit_time, title, xlabel, ylabel, savefile)

def plotSilhouette(k, X, dataset, random_seed):
    print('Plotting k=%s for dataset %s ' % (k, dataset))
    fig, ax1 = plt.subplots()
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(X) + (k + 1) * 10])
    clf = GaussianMixture(n_components=k, random_state=random_seed)
    cluster_labels = clf.fit_predict(X)
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
    ax1.set_title('EM Silhouette Plot ' + dataset)
    ax1.set_ylabel('Cluster')
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_yticks([])
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    if dataset == 'car':
        savefile = 'plots/sil_car/EM_Silhouette_Plot_' + str(k) + '_' + dataset + '.png'
    else:
        savefile = 'plots/sil_adult/EM_Silhouette_Plot_' + str(k) + '_' + dataset + '.png'
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
    k = np.arange(2, 51)
    gmm(k, X_1_scaled, y_1, dataset_1, random_seed)
    plotSilhouette(2, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(3, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(4, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(9, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(10, X_1_scaled, dataset_1, random_seed)
    plotSilhouette(11, X_1_scaled, dataset_1, random_seed)
    gmm(k, X_2_scaled, y_2, dataset_2, random_seed)
    plotSilhouette(2, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(3, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(4, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(7, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(8, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(9, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(10, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(13, X_2_scaled, dataset_2, random_seed)
    plotSilhouette(14, X_2_scaled, dataset_2, random_seed)
    '''
    car - cluster = 4
    adult - cluster = 2, 3, 4
    '''
    clf_1 = GaussianMixture(n_components=4, random_state=random_seed)
    clf_2 = GaussianMixture(n_components=4, random_state=random_seed)
    clf_1.fit(X_1_scaled)
    clf_2.fit(X_2_scaled)
    cluster_labels_1 = clf_1.fit_predict(X_1_scaled)
    cluster_labels_1 = pd.DataFrame(data=cluster_labels_1, columns=['Labels'])
    cluster_labels_1_counts = cluster_labels_1['Labels'].value_counts()
    cluster_labels_2 = clf_2.fit_predict(X_2_scaled)
    cluster_labels_2 = pd.DataFrame(data=cluster_labels_2, columns=['Labels'])
    cluster_labels_2_counts = cluster_labels_2['Labels'].value_counts()
    title, xlabel, ylabel = ['KMeans - ' + dataset_1, 'Label', 'Count']
    savefile = 'plots/EM_Label_' + dataset_1 + '.png'
    plotBar(cluster_labels_1_counts, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['KMeans - ' + dataset_2, 'Label', 'Count']
    savefile = 'plots/EM_Label_' + dataset_2 + '.png'
    plotBar(cluster_labels_2_counts, title, xlabel, ylabel, savefile)
