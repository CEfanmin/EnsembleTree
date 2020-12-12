import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score


# load dataset and preprocessing
dpath = './data/'
data = pd.read_csv(dpath + "mushrooms.csv")
from sklearn.preprocessing import LabelEncoder
labebelencoder = LabelEncoder()
for col in data.columns:
	data[col] = labebelencoder.fit_transform(data[col])
X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
'''
#Logistic Regression model
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression(n_jobs=2)
model_LR.fit(X_train, y_train)
y_prob = model_LR.predict_proba(X_test)[:,1]
y_pred = np.where(y_prob >0.5, 1, 0)
# print (model_LR.score(X_test, y_pred))
auc_roc = metrics.roc_auc_score(y_test, y_pred)
print ("before logistic tuned model auc: ", auc_roc)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
LR_model = LogisticRegression()
tuned_parameters = {"C":[0.01, 0.1, 1, 10, 100],
					"penalty":['l1', 'l2']
					}
from sklearn.model_selection import GridSearchCV
import time 
LR = GridSearchCV(LR_model, tuned_parameters, cv=10, n_jobs=2)
LR.fit(X_train, y_train)
print("LR.best_params_: ",LR.best_params_)
y_prob = LR.predict_proba(X_test)[:,1] 
y_pred = np.where(y_prob > 0.5, 1, 0) 
print(LR.score(X_test, y_pred))
auc_roc = metrics.roc_auc_score(y_test, y_pred)
print("after logistic tune auc: ", auc_roc)
'''

'''
# decision tree
from sklearn.tree  import DecisionTreeClassifier
mode_tree = DecisionTreeClassifier()
mode_tree.fit(X_train, y_train)
y_prob = LR.predict_proba(X_test)[:,1] 
y_pred = np.where(y_prob > 0.5, 1, 0) 
print(LR.score(X_test, y_pred))
auc_roc = metrics.roc_auc_score(y_test, y_pred)
print("before decision tree tune auc: ",auc_roc)

tuned_parameters = {'max_features':["auto", "log2"],
					'min_samples_leaf':range(1,10,2),
					'max_depth':range(2,5,1)	
					}

DD = GridSearchCV(mode_tree, tuned_parameters, cv=5)
DD.fit(X_train, y_train)
print("DD.best_params_: ", DD.best_params_)
y_prob = DD.predict_proba(X_test)[:,1] 
y_pred = np.where(y_prob > 0.5, 1, 0) 
DD.score(X_test, y_pred)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
print("after decision tree tune auc: ",auc_roc)
'''
from xgboost import XGBClassifier
model_XGB = XGBClassifier()
model_XGB.fit(X_train, y_train)
y_prob = model_XGB.predict_proba(X_test)[:,1] 
y_pred = np.where(y_prob > 0.5, 1, 0) 
model_XGB.score(X_test, y_pred)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
print("before xgboost tune auc: ",auc_roc)
print("feature importance: ", model_XGB.feature_importances_)

from matplotlib import pyplot
pyplot.bar(range (len(model_XGB.feature_importances_)), model_XGB.feature_importances_)
pyplot.show()

from xgboost import plot_importance
plot_importance(model_XGB)
pyplot.show()

from numpy import sort
from sklearn.feature_selection import SelectFromModel

thresholds = sort(model_XGB.feature_importances_)
for thresh in thresholds:
	selection = SelectFromModel(model_XGB, threshold= thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
  	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" %(thresh, select_X_train.shape[1],
	  accuracy*100.0))

