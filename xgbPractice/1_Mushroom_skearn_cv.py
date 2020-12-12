from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
#from sklearn.model_selection import KFold

my_workpath = './data/'
X_train,y_train = load_svmlight_file(my_workpath + 'agaricus.txt.train')
X_test,y_test = load_svmlight_file(my_workpath + 'agaricus.txt.test')

bst =XGBClassifier(max_depth=2, learning_rate=0.1,n_estimators=100, 
                   silent=True, objective='binary:logistic')

# stratified k-fold cross validation evaluation of xgboost model
#kfold = KFold(n_splits=10, random_state=7)
kfold = StratifiedKFold(n_splits=10, random_state=7)
#fit_params = {'eval_metric':"logloss"}
#results = cross_val_score(bst, X_train, y_train, cv=kfold, fit_params)
results = cross_val_score(bst, X_train, y_train, cv=kfold)
print results
print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))