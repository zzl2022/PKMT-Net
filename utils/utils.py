
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from seaborn import heatmap
from sklearn.preprocessing import OneHotEncoder


def data_preprocess(data_path):

    Training_set = pd.read_csv(data_path+'/Training_set.csv')
    Test_set = pd.read_csv(data_path+'/Test_set.csv')
    External_Test_set = pd.read_csv(data_path+'/External_Test_set.csv')

    data_training = Training_set.iloc[:, 1:-1]
    label_training = Training_set.iloc[:, -1]
    data_test = Test_set.iloc[:, 1:-1]
    label_test = Test_set.iloc[:, -1]
    data_test_external = External_Test_set.iloc[:, 1:-1]
    label_test_external = External_Test_set.iloc[:, -1]

    scaler = StandardScaler()
    data_training_scaled = scaler.fit_transform(data_training)
    data_test_scaled = scaler.transform(data_test)
    data_test_external_scaled = scaler.transform(data_test_external)

    return data_training_scaled, data_test_scaled, data_test_external_scaled,\
           label_training, label_test, label_test_external


def selected_features(all_features, best_agent):

    best_agent = pd.DataFrame(best_agent)
    result = pd.concat([all_features, best_agent], axis=1)
    result.columns = ['all features', 'selected']

    features_selected = result[result['selected'] == 1]['all features'].rename('features')

    return features_selected


def model_test(model, data, label, best_agent, flag):

    cols = np.flatnonzero(best_agent)

    X_test = np.array(data)
    test_data = X_test[:, cols]
    y_pred = model.predict(test_data)
    y_prob = model.predict_proba(test_data)  # 得到正类的概率值

    if flag == 'train':
        with pd.ExcelWriter('./result/output.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            pd.DataFrame(label).to_excel(writer, index=False, sheet_name='train_label')
            pd.DataFrame(y_prob).to_excel(writer, index=False, sheet_name='train_y_prob')
    elif flag == 'test':
        with pd.ExcelWriter('./result/output.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            pd.DataFrame(label).to_excel(writer, index=False, sheet_name='test_label')
            pd.DataFrame(y_prob).to_excel(writer, index=False, sheet_name='test_y_prob')
    else:
        with pd.ExcelWriter('./result/output.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            pd.DataFrame(label).to_excel(writer, index=False, sheet_name='test_external_label')
            pd.DataFrame(y_prob).to_excel(writer, index=False, sheet_name='test_external_y_prob')

    # 输出分类报告
    report = classification_report(label, y_pred)
    print("Classification Report:\n", report)

    # encoder = OneHotEncoder(sparse=False, categories='auto')
    # label = encoder.fit_transform(label.reshape(-1, 1))
    # auc = roc_auc_score(label, y_prob, average='macro')

    auc = roc_auc_score(label, y_prob[:, 1])

    print(f"AUC: {auc:.4f}")


def evaluation_metrics(true_label, pre_score):

    df = pd.DataFrame()

    pre_score = pd.DataFrame(pre_score)
    pre_label = np.argmax(pre_score.values, axis=1)

    auc = metrics.roc_auc_score(true_label, pre_score.iloc[:, 1])

    accuracy = metrics.accuracy_score(true_label, pre_label)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_label, pre_label)

    tn, fp, fn, tp = conf_matrix.ravel()

    # 计算敏感性（Sensitivity/Recall）
    sensitivity = tp / (tp + fn)
    # 计算特异性（Specificity）
    specificity = tn / (tn + fp)

    # 计算正预测值（Positive Predictive Value/ Precision）
    ppv = tp / (tp + fp)
    # 计算负预测值（Negative Predictive Value）
    npv = tn / (tn + fn)

    # # 计算正似然比（Positive Likelihood Ratio）
    # plr = sensitivity / (1 - specificity)
    # # 计算负似然比（Negative Likelihood Ratio）
    # nlr = (1 - sensitivity) / specificity
    # df.loc[0, 'fitness'] = "{:.4f}".format(fitness)
    df.loc[0, 'auc'] = f'{auc:.4f}'
    # df.loc[0, 'num'] = np.sum(agent[0])

    df.loc[0, 'accuracy'] = "{:.4f}".format(accuracy)

    df.loc[0, 'sensitivity'] = "{:.4f}".format(sensitivity)
    df.loc[0, 'specificity'] = "{:.4f}".format(specificity)

    df.loc[0, 'ppv'] = "{:.4f}".format(ppv)
    df.loc[0, 'npv'] = "{:.4f}".format(npv)

    # df.loc[0, 'plr'] = "{:.4f}".format(plr)
    # df.loc[0, 'nlr'] = "{:.4f}".format(nlr)

    return df


class Result():
    # structure of the result
    def __init__(self):
        self.ranks = None
        self.scores = None
        self.features = None
        self.ranked_features = None


def PasiLuukka(in_data, target, measure='luca', p=1):
    d = pd.DataFrame(in_data)
    t = pd.DataFrame(target)
    data = pd.concat([d, t], axis=1)

    # Feature selection method using similarity measure and fuzzy entropy
    # measures based on the article:
    # P. Luukka, (2011) Feature Selection Using Fuzzy Entropy Measures with
    # Similarity Classifier, Expert Systems with Applications, 38, pp. 4600-4607

    l = int(max(data.iloc[:, -1]))
    m = data.shape[0]
    t = data.shape[1] - 1

    dataold = data.copy()

    idealvec_s = np.zeros((l, t))
    for k in range(l):
        idx = data.iloc[:, -1] == k + 1
        idealvec_s[k, :] = data[idx].iloc[:, :-1].mean(axis=0)

    # scaling data between [0,1]
    data_v = data.iloc[:, :-1]
    data_c = data.iloc[:, -1]  # labels
    mins_v = data_v.min(axis=0)
    Ones = np.ones((data_v.shape))
    data_v = data_v + np.dot(Ones, np.diag(abs(mins_v)))

    tmp = []
    for k in range(l):
        tmp.append(abs(mins_v))

    idealvec_s = idealvec_s + tmp
    maxs_v = data_v.max(axis=0)
    data_v = np.dot(data_v, np.diag(maxs_v ** (-1)))
    tmp2 = [];
    for k in range(l):
        tmp2.append(abs(maxs_v))

    idealvec_s = idealvec_s / tmp2

    data_vv = pd.DataFrame(data_v)  # Convert the array of feature to a dataframe
    data = pd.concat([data_vv, data_c], axis=1, ignore_index=False)

    # sample data
    datalearn_s = data.iloc[:, :-1]

    # similarities
    sim = np.zeros((t, m, l))

    for j in range(m):
        for i in range(t):
            for k in range(l):
                sim[i, j, k] = (1 - abs(idealvec_s[k, i] ** p - datalearn_s.iloc[j, i]) ** p) ** (1 / p)

    sim = sim.reshape(t, m * l)

    # possibility for two different entropy measures
    if measure == 'luca':
        # moodifying zero and one values of the similarity values to work with
        # De Luca's entropy measure
        delta = 1e-10
        sim[sim == 1] = delta
        sim[sim == 0] = 1 - delta
        H = (-sim * np.log(sim) - (1 - sim) * np.log(1 - sim)).sum(axis=1)
    elif measure == 'park':
        H = (np.sin(np.pi / 2 * sim) + np.sin(np.pi / 2 * (1 - sim)) - 1).sum(axis=1)

    feature_values = np.array(in_data)
    result = Result()
    result.features = feature_values
    result.scores = H
    result.ranks = np.argsort(np.argsort(-H))
    result.ranked_features = feature_values[:, result.ranks]
    return result


def plot_confusion_matrix(true_label, pre_score, path, set):

    pre_label = np.argmax(pre_score, axis=1)

    # 自定义颜色映射
    # colors = [(92 / 255, 141 / 255, 196 / 255), (209 / 255, 162 / 255, 184 / 255)]  # 自定义两种颜色
    colors = [(92 / 255, 141 / 255, 196 / 255), (236 / 255, 205 / 255, 216 / 255)]  # 自定义两种颜色

    n_bins = 100  # 设定颜色梯度的细致程度
    cmap_name = "custom_heatmap"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    plt.figure(figsize=(1.7, 1.8), dpi=300)

    confusion_mat = metrics.confusion_matrix(true_label, pre_label)
    accuracy_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    # heatmap(confusion_mat, annot=False, cmap=cmap, cbar=False, yticklabels=['NT', 'ACA', 'SCC'],
    #         xticklabels=['NT', 'ACA', 'SCC'])

    heatmap(confusion_mat, annot=False, cmap=cmap, cbar=False, yticklabels=['NT', 'ACA'],
            xticklabels=['NT', 'ACA'])

    # 在每个单元格中心显示正确率并设置字体大小
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat[0])):
            num = round(confusion_mat[i][j], 4)
            plt.text(j + 0.5, i + 0.5, num, fontsize=8, ha="center", va="center")

    # plt.title("Confusion matrix for test-dataset")
    plt.xlabel("Predicted labels", fontsize=8, fontname='Arial')
    plt.ylabel("True labels", fontsize=8, fontname='Arial')
    plt.xticks(fontsize=6, fontname='Arial')
    plt.yticks(fontsize=6, fontname='Arial')

    if set == 'train':
        plt.title('Training set', fontsize=8, fontname='Arial')
    elif set == 'val':
        plt.title('Validation set', fontsize=8, fontname='Arial')
    else:
        plt.title('Test set', fontsize=8, fontname='Arial')

    plt.tight_layout()
    plt.savefig(path + set + '_confusion_matrix.png', dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close()
