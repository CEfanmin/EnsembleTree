import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

my_workpath = './data/'
X_train,y_train = load_svmlight_file(my_workpath + 'agaricus.txt.train')
X_test,y_test = load_svmlight_file(my_workpath + 'agaricus.txt.test')

seed =7
test_size =0.33
X_train_part, X_validate, y_train_part, y_validate = train_test_split(X_train,y_train,
	test_size= test_size, random_state =seed)

num_round = 100
bst =XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round, 
					silent=True, objective='binary:logistic')
eval_set = [(X_train_part, y_train_part), (X_validate, y_validate)]
bst.fit(X_train_part, y_train_part, eval_metric = ["error", "logloss"], 
	eval_set=eval_set, verbose = False)


results = bst.evals_result()
# epochs = len(results['validation_0']['error'])
x_axis = range(0, num_round)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()

# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()


