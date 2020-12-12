# classifier -> fit -> predict
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

my_workpath = './data/'
X_train,y_train = load_svmlight_file(my_workpath + 'agaricus.txt.train')
X_test,y_test = load_svmlight_file(my_workpath + 'agaricus.txt.test')
bst =XGBClassifier(max_depth=2, learning_rate=0.1, silent=True, 
					objective='binary:logistic')

param_test = {
 'n_estimators': range(1, 51, 1)
}

clf = GridSearchCV(estimator = bst,n_jobs=2, param_grid = param_test, scoring='accuracy', cv=5)
clf.fit(X_train, y_train)
print (clf.grid_scores_, clf.best_params_, clf.best_score_)

preds = clf.predict(X_test)
predictions = [round(value) for value in preds]
test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy of gridsearchcv: %.2f%%" % (test_accuracy * 100.0))
		


