import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

my_workpath = './data/'
X_train,y_train = load_svmlight_file(my_workpath + 'agaricus.txt.train')
X_test,y_test = load_svmlight_file(my_workpath + 'agaricus.txt.test')

#split data
seed =7
test_size =0.33
X_train_part, X_validate, y_train_part, y_validate = train_test_split(X_train,y_train,
	test_size= test_size, random_state =seed)

num_round = 100
bst =XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round, 
					silent=True, objective='binary:logistic')

# early_stopping_rounds=10 means:stop if there is no improve in 10 iterations 
eval_set = [(X_validate, y_validate)]
bst.fit(X_train_part, y_train_part, early_stopping_rounds=10, eval_metric="error",
    eval_set=eval_set, verbose=True)

# retrieve performance metrics
results = bst.evals_result()
#print(results)

epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Test')
ax.legend()
pyplot.ylabel('Error')
pyplot.xlabel('Round')
pyplot.title('XGBoost Early Stop')
pyplot.show()
