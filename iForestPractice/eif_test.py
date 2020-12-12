#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eif_test.py
@Time    :   2019/09/16 15:43:01
@Author  :   fanmin 
'''

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
import eif as iso
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import time,sys
sys.dont_write_bytecode = True  # no pyc file
from data_explore import load_data


def getDepth(x, root, d):
    n = root.n
    p = root.p
    if root.ntype == 'exNode':
        return d
    else:
        if (x-p).dot(n) < 0:
            return getDepth(x,root.left,d+1)
        else:
            return getDepth(x,root.right,d+1)

def getVals(forest, x, sorted=True):
    theta = np.linspace(0,2*np.pi, forest.ntrees)
    r = []
    for t in forest.Trees:
        r.append(getDepth(x, t.root, 1))
    if sorted:
        r = np.sort(np.array(r))
    return r, theta

def visualize(forest, dims):
    Sorted=False
    fig = plt.figure(figsize=(12,6))
    ax1 = plt.subplot(111, projection='polar')

    rn, thetan = getVals(forest, np.zeros(dims), sorted=Sorted)
    for j in range(len(rn)):
        ax1.plot([thetan[j], thetan[j]], [1, rn[j]], color='b',alpha=1,lw=1)

    ra, thetaa = getVals(forest, 3.3+np.zeros(dims), sorted=Sorted)
    for j in range(len(ra)):
        ax1.plot([thetaa[j], thetaa[j]], [1, ra[j]], color='r',alpha=0.9,lw=1.3)
        
    ax1.set_title("Nominal: Mean={0:.3f}, Var={1:.3f}\nAnomaly: Mean={2:.3f}, Var={3:.3f}".format(np.mean(rn),np.var(rn),np.mean(ra),np.var(ra)))
    
    ax1.set_xticklabels([])
    ax1.set_xlabel("Anomaly")
    ax1.set_ylim(0, forest.limit)

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = load_data()
    print("training sample nums: ",len(X_train))

    eifmodel = iso.iForest(X_train.values, ntrees=100, sample_size=256, ExtensionLevel=1, n_jobs=4)
    
    #  save model
    joblib.dump(eifmodel, './eiforest.pkl')
    # eifmodel = joblib.load('./eiforest.pkl')
    print("test sample nums: ", len(X_test))
    print("test anoamly sample nums: ", sum(y_test))

    stime = time.time()
    y_pred_test = eifmodel.compute_paths(X_test.values, n_jobs=4)
    ctime = time.time() - stime
    print("cost time is: {:.4f} ".format(ctime))
    fpr, tpr, thresholds = roc_curve(y_test.values, y_pred_test)
    areaUnderROC = auc(fpr, tpr)
    print("Test set AUC:{:.4f} ".format(areaUnderROC))
    
    # pd.DataFrame(y_pred_test).to_csv("./score.csv", header=False)
    # print(np.transpose(confusion_matrix(y_test.values, y_pred_test,labels=[-1,1])))
    # print(classification_report(y_test.values,y_pred_test, digits=4))

    # visualize(eifmodel, dims=28)
    