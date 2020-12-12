from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from matplotlib import pyplot
import seaborn as sns

dpath = './data/'
train = pd.read_csv(dpath + 'Otto_train.csv')
train.head(10)
sns.countplot(train.target)
pyplot.xlabel('target')
pyplot.ylabel('Number of occurrences')

#drop ids and get labels
y_train = train['target']
y_train = y_train.map(lambda s: s[6:])
y_train = y_train.map(lambda s: int(s)-1)
train =train.drop(['id', 'target'], axis=1)
X_train = np.array(train)
kfold = StratifiedKFold(n_splits=5, shuffle =True, random_state=3)

def modelfit(alg,X_train,y_train,useTrainCV=True,cv_folds =None,early_stopping_rounds=50):
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgb_param['num_class']=9
		xgtrain = xgb.DMatrix(X_train, label = y_train)
		cvresult = xgb.cv(xgb_param,xgtrain, num_boost_round=alg.get_params()['n_estimators']
			, folds = cv_folds)
		n_estimators = cvresult.shape[0]
		alg.set_params(n_estimators = n_estimators)
		print cvresult
		cvresult.to_csv('my_preds_4_1.csv',index_label='n_estimators')
		test_means = cvresult['test-mlogloss-mean']
		test_stds = cvresult['test-mlogloss-std']
		train_means = cvresult['train-mlogloss-mean']
		train_stds = cvresult['train-mlogloss-std']
		x_axis = range(0, n_estimators)
		pyplot.errorbar(x_axis, test_means, yerr = test_stds, label='Test')
		pyplot.errorbar(x_axis,train_means, yerr = train_stds, label='Train')
		pyplot.title("XGBoost n_estimators vs Log loss")
		pyplot.xlabel("n_estimators")
		pyplot.ylabel("Log loss")
		pyplot.savefig("n_estimators.png")

	alg.fit(X_train, y_train, eval_metric='mlogloss')
	train_predprob = alg.predict_proba(X_train)
	logloss = log_loss(y_train, train_predprob)
	print("logloss of train")
	print (logloss)

xgb1 = XGBClassifier(learning_rate =0.1, n_estimators=1000,
					max_depth=5, min_child_weight=1, gamma=0,
					subsample=0.3, colsample_bytree=0.8,
					colsample_bylevel=0.7, objective ='multi:softprob',seed =3)

modelfit(xgb1, X_train, y_train,cv_folds=kfold)

# plot csv
cvresult = pd.DataFrame.from_csv('my_preds_4_1.csv')
test_means = cvresult['test-mlogloss-mean']
test_stds = cvresult['test-mlogloss-std'] 
train_means = cvresult['train-mlogloss-mean']
train_stds = cvresult['train-mlogloss-std'] 
x_axis = range(0, cvresult.shape[0])        
pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
pyplot.title("XGBoost n_estimators vs Log Loss")
pyplot.xlabel( 'n_estimators' )
pyplot.ylabel( 'Log Loss' )
pyplot.savefig( 'n_estimators4_1.png' )
pyplot.show()


cvresult = pd.DataFrame.from_csv('my_preds_4_1.csv')
cvresult = cvresult.iloc[100:]
# plot
test_means = cvresult['test-mlogloss-mean']
test_stds = cvresult['test-mlogloss-std'] 
train_means = cvresult['train-mlogloss-mean']
train_stds = cvresult['train-mlogloss-std'] 
x_axis = range(100,cvresult.shape[0]+100)
fig = pyplot.figure(figsize=(10, 10), dpi=100)
pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
pyplot.title("XGBoost n_estimators vs Log Loss")
pyplot.xlabel( 'n_estimators' )
pyplot.ylabel( 'Log Loss' )
pyplot.savefig( 'n_estimators_detail.png' )
pyplot.show()
