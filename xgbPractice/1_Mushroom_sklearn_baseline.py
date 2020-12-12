from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

my_workpath = './data/'
X_train,y_train = load_svmlight_file(my_workpath + 'agaricus.txt.train')
X_test,y_test = load_svmlight_file(my_workpath + 'agaricus.txt.test')

print (X_train.shape)
print (X_test.shape)

num_round =2 #interations
bst = XGBClassifier(max_depth=2,learning_rate=1,n_estimators=num_round,
					silent=True, objective='binary:logistic')
#predictions train
bst.fit(X_train, y_train)
train_preds = bst.predict(X_train)
train_predictions=[round(value) for value in train_preds]
train_accuracy = accuracy_score(y_train, train_predictions)
print("Train Accuracy: %.2f" %train_accuracy)

#predictions test 
preds = bst.predict(X_test)
predictions = [round(value) for value in preds]
test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f" %test_accuracy)
