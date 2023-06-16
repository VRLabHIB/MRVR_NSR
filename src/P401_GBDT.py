import numpy as np
import pandas as pd
import os
import seaborn as sns
import glob
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import shap
import seaborn as sns
from pathlib import Path

import os


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def GBDT(source, target, method, n_iter=50, n_estimators=100):
    # method could also be binary
    print('Target variable', target.columns.tolist())
    print('Source variables', source.columns.tolist())
    target = target.values.ravel()

    sc_X = StandardScaler()
    X = sc_X.fit_transform(source)

    y = np.where(target == 3, int(1), int(-1))

    acc_lst = list()
    f1_lst = list()
    quickprec_lst = list()
    slowprec_lst = list()

    best = 0

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    shap_values = np.zeros(X_test.shape)

    for i in range(n_iter):
        print(i)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

        gbm = ensemble.GradientBoostingClassifier(n_estimators=n_estimators)
        gbm.fit(X_train, Y_train)

        prediction = gbm.predict(X_test)
        accuracy = accuracy_score(Y_test, prediction)
        acc_lst.append(accuracy)
        if method == 'median':
            f1 = f1_score(Y_test, prediction, labels=["2D", "3D"])
            f1_lst.append(f1)
        if method == 'binary':
            f1 = f1_score(Y_test, prediction, labels=["2D", "3D"])
            f1_lst.append(0)

        qprecision = precision_score(Y_test, prediction, pos_label=-1)
        quickprec_lst.append(qprecision)
        sprecision = precision_score(Y_test, prediction, pos_label=1)
        slowprec_lst.append(sprecision)

        shap_values = shap_values + shap.TreeExplainer(gbm).shap_values(X_test)

        if accuracy > best:
            gbmb = gbm
            Y_testb = Y_test
            predictionb = prediction
            X_testb = X_test
            best = accuracy

    shap_value = shap.TreeExplainer(gbmb).shap_values(X_testb)
    shap_values = shap_values / n_iter
    contingency = ((np.abs(Y_testb - predictionb) * 0.5) + 1) % 2

    return [X_testb, Y_testb, predictionb, best, acc_lst, shap_values, quickprec_lst, slowprec_lst, f1_lst, contingency,
            shap_value]


from sklearn.metrics import confusion_matrix


def plotting(target, values, labels, shap_values, shap_value, features, generalized, acc_lst, best_acc, name,
             columns, project_path):
    plt.rcParams.update({'font.size': 12})
    plt.figure(tight_layout=True, figsize=(20, 15))

    plt.subplot(2, 2, 2)
    plt.text(0.5, 0.5,
             'Model Performance\n \n Features: {} \n Generalized: {}\n \n Accuracy: Mean {}, SD {}\n Best Model accuracy {}'.format(
                  features, generalized, np.round(np.nanmean(acc_lst), 3), np.round(np.nanstd(acc_lst), 3),
                 np.round((best_acc * 100), 3)),
             ha='center', va='center', size=16)
    plt.axis('off')
    plt.subplot(2, 2, 1)
    mat = confusion_matrix(target, values)
    sns.heatmap(data=mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title('Confusion Matrix for the Best Model')

    plt.subplot(2, 2, 3)
    shap.initjs()
    shap.summary_plot(shap_values, X_test, feature_names=columns, plot_type="bar", plot_size=None, show=False)
    w, _ = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(w, w * 2.5 / 4)
    plt.tight_layout()
    plt.subplot(2, 2, 4)
    shap.summary_plot(shap_value, X_test, feature_names=columns, plot_type='dot', plot_size=None, show=False)
    w, _ = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(w, w * 2.5 / 4)
    plt.tight_layout()
    path = project_path+ '\\results\\'

    plt.savefig(path + '{}.jpg'.format(name), bbox_inches='tight', dpi=500)
    plt.show()

if __name__ == '__main__':
    val_path = os.path.abspath(os.getcwd())
    project_path = os.path.abspath(Path(val_path).parent)
    data_path = project_path + '\\data\\6_feature_dataset\\'

    df = pd.read_csv(data_path + '2023-06-16_eye_features.csv')
    print(len(df))

    df = df.dropna(axis = 'index', how = 'any', ignore_index = True)
    #df = df.drop(columns=['Equal fixation duration within figure', 'Mean regressive fixation duration'])
    print(len(df))

    import random

    random.seed(10092022)

    df = df.reset_index(drop=True)


    source = df.iloc[:, 7:-3]

    target = df[['dimension']].astype(int)

    model = GBDT(source, target, n_iter=100, n_estimators=100, method='median')

    # model_writer('self3D', model)

    columns = df.columns[7:-3]

    # model = model_reader('2D3D')
    X_test = model[0]
    Y_test = model[1]
    prediction = model[2]
    best_acc = model[3]
    acc_lst = model[4]
    shap_values = model[5]
    quickprec_lst = model[6]
    slowprec_lst = model[7]
    f1s = model[8]
    contingency = list(map(bool, model[9]))
    shap_value = model[10]

    print('Mean Acc: ', np.round(np.nanmean(acc_lst), 3), 'SD: ', np.round(np.nanstd(acc_lst), 3))
    print('Best Model accuracy ', np.round(best_acc, 3))
    print('Quick Precision: {}'.format(np.round(np.nanmean(quickprec_lst), 3)))
    print('Slow Precision: {}'.format(np.round(np.nanmean(slowprec_lst), 3)))

    features = '2D3D'
    split = 'binary'
    plotting(Y_test, prediction, ["2D", "3D"], shap_values, shap_value, features, split, acc_lst, best_acc,
             features,  columns, project_path)




