import numpy as np
import xgboost as xgb

test_size = 550000
dpath = './data/'
dtrain = np.loadtxt( dpath+'/higgsboson_training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
print ('finish loading from csv ')

label  = dtrain[:,32]
data   = dtrain[:,1:31]
# rescale weight to make it same as test set
weight = dtrain[:,31] * float(test_size) / len(label)

sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

# print weight statistics
print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))

dtrain =xgb.DMatrix(data, label=label, missing=-999.0,weight = weight)
param = {}
param['objective'] = 'binary:logitraw'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1

# boost 1000 trees
num_round = 1000

print ('running cross validation, with preprocessing function')
# define the preprocessing function
# used to return the preprocessed training, test data, and parameter
# we can use this to do weight rescale, etc.
# as a example, we try to set scale_pos_weight
def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()
    sum_weight = sum(wtrain) + sum(wtest)
    wtrain *= sum_weight / sum(wtrain)
    wtest *= sum_weight / sum(wtest)
    dtrain.set_weight(wtrain)
    dtest.set_weight(wtest)
    return (dtrain, dtest, param)

# do cross validation, for each fold
# the dtrain, dtest, param will be passed into fpreproc
# then the return value of fpreproc will be used to generate
# results of that fold
cvresult = xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'ams@0.15', 'auc'}, early_stopping_rounds=10, seed = 0, fpreproc = fpreproc)

print ('finish cross validation')

n_estimators = cvresult.shape[0]
from matplotlib import pyplot
# plot
test_means = cvresult['test-ams@0.15-mean']
test_stds = cvresult['test-ams@0.15-std'] 
        
train_means = cvresult['train-ams@0.15-mean']
train_stds = cvresult['train-ams@0.15-std'] 

x_axis = range(0, n_estimators)
pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
pyplot.title("HiggsBoson n_estimators vs ams@0.15")
pyplot.xlabel( 'n_estimators' )
pyplot.ylabel( 'ams@0.15' )
pyplot.savefig( 'HiggsBoson_estimators.png' )
pyplot.show()

print('train model using the best parameters by cv...')
bst = xgb.train(param, dtrain, n_estimators)
bst.save_model('higgs_cv.model')
print('train finished')

