from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

my_workpath = './data/'
X_train,y_train = load_svmlight_file(my_workpath + 'agaricus.txt.train')
X_test,y_test = load_svmlight_file(my_workpath + 'agaricus.txt.test')

#split data
seed =7
test_size = 0.33
X_train_part, X_validate, y_train_part, y_validate = train_test_split(X_train, y_train,
	test_size = test_size)

num_round =2 #interations
bst = XGBClassifier(max_depth=2,learning_rate=1,n_estimators=num_round,
					silent=True, objective='binary:logistic')
bst.fit(X_train_part, y_train_part)
validare_preds = bst.predict(X_validate)

validate_predictions = [round(value) for value in validare_preds]
trian_accuracy = accuracy_score(y_validate, validate_predictions)
print ("Trian Accuracy: %.2f" %trian_accuracy)

preds = bst.predict(X_test)
predictions = [round(value) for value in preds]
test_accuracy =accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f"%test_accuracy)
