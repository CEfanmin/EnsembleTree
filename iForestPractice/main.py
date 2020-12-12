#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2019/09/11 09:37:21
@Author  :   fanmin 
'''

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time, sys
sys.dont_write_bytecode = True  # no pyc file
from data_explore import load_data


def plot_figure(fpr, tpr, areaUnderROC):
	plt.figure()
	plt.plot(fpr, tpr, color='g', lw=2, label='CDCNN ROC')
	plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(' Area under the curve = {0:0.2f}'.format(areaUnderROC))
	plt.legend(loc="lower right")
	plt.show()

def visualize(X_test, y_pred_test):
    # TSNE transform
    # pca_tsne = TSNE(n_components=2, random_state=14)
    # X_test = pd.DataFrame(
    #     pca_tsne.fit_transform(X_test),
    #     columns=['V'+str(x) for x in range(1, 3)]
    # )

    X_test['label'] = y_pred_test
    print(X_test.columns)
    # visualize
    categories = np.unique(y_pred_test)
    colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]
    for i, category in enumerate(categories):
        plt.scatter('V1', 'V2', data=X_test.loc[X_test.label==category, :],
                                s=20, cmap=colors[i], marker='+',
                                label=str(category)
                    )
    
    plt.gca().set(xlabel='V1', ylabel='V2')
    plt.title("anomaly results")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = load_data()
    print("training sample nums: ",len(X_train))
    # fit the model
    imodel = IsolationForest(n_estimators=100, max_samples=256, random_state=2019,n_jobs=-1,
                            bootstrap=False)
    imodel.fit(X_train)
    # # save model
    joblib.dump(imodel, './iforest.pkl')

    # predict 
    # imodel = joblib.load('./iforest.pkl')
    stime = time.time()
    y_pred_test = imodel.predict(X_test.values)
    print("test sample nums: ", len(X_test))
    ctime = time.time() - stime
    print("cost time is:{:.4f} ".format(ctime))
    y_test[y_test==1] = -1
    y_test[y_test==0] = 1
    
    fpr, tpr, thresholds = roc_curve(y_test.values, y_pred_test)
    areaUnderROC = auc(fpr, tpr)
    print("Test set AUC:{:.4f} ".format(areaUnderROC)) 

    print(np.transpose(confusion_matrix(y_test.values, y_pred_test,labels=[-1,1])))
    print(classification_report(y_test.values,y_pred_test, digits=4))
    
    # plot_figure(fpr, tpr, areaUnderROC)
    # visualize(X_test, y_pred_test)
    # visualize(X_test, y_test.values)

