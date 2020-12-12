import xgboost as xgb
from xgboost import plot_tree
from sklearn.metrics import accuracy_score


my_workspace = './data/'
dtrain = xgb.DMatrix(my_workspace + 'agaricus.txt.train')
dtest = xgb.DMatrix(my_workspace + 'agaricus.txt.test')

# specify parameters via map
param = {'max_depth':4, 'eta':1, 'silent':0, 'objective':'binary:logistic'}

# boosting itera
num_round =2   # tree nums
bst = xgb.train(param, dtrain, num_round) # bst is trian odel

# train prediction
train_preds = bst.predict(dtrain)
train_predictions = [round(value) for value in train_preds]
y_train = dtrain.get_label()
train_AUC= accuracy_score(y_train, train_predictions)
print("trian_AUC: %.2f" %train_AUC)

#test prediction
preds = bst.predict(dtest)
predictions = [round(value) for value in preds]
y_test = dtest.get_label()
test_AUC = accuracy_score(y_test, predictions)
print("test_AUC: %.2f%%" % (test_AUC*100))

# plot tree
import graphviz
from matplotlib import pyplot
xgb.plot_tree(bst, num_trees=1, rankdir='UT')
pyplot.show()



