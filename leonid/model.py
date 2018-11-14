# coding: utf-8
import sys
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import time

from scipy import stats
from scipy.sparse import hstack, csr_matrix, load_npz, save_npz
from sklearn.model_selection import train_test_split

from collections import Counter
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
#from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
stop = set(stopwords.words('russian'))
import lightgbm as lgb
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold
import gc

try:
    import cPickle as pickle
except:
    import pickle

#import shap


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


pd.set_option('max_columns', 60)


RANDOM_STATE = 3542
np.random.seed(RANDOM_STATE)


FILENO = 24

sub = pd.read_csv('../input/sample_submission.csv', index_col = 'item_id')
holdout_index = pd.read_csv('../data/holdout__index_itemid_v2.csv', index_col = 'item_id')
holdout_train_index = pd.read_csv('../data/holdout_train__index_itemid_v2.csv', index_col = 'item_id')#, index_col = 'index'
holdout_train_folds_all_index = pd.read_csv('../data/holdout_train_folds_all__index_itemid_v2.csv')
train_folds_all_index = pd.read_csv('../data/train_folds_all__index_itemid_v2.csv', index_col = 'item_id')
#####################

train = pd.read_csv('../input/train.csv', index_col = 'item_id')#, parse_dates = ['activation_date']
traindex = train.index
test = pd.read_csv('../input/test.csv', index_col = 'item_id')#, parse_dates = ['activation_date']
testdex = test.index

##############################
train_store = './train_with_features_514_no_meta.pkl'
test_store = './test_with_features_514_no_meta.pkl'
if os.path.isfile(train_store):
    print("loading train data from pickle file", train_store)
    with open(os.path.abspath(train_store), 'rb') as f:
        train_with_features = pickle.load(f, encoding='bytes')
        train_with_features.drop(['image'], axis=1, inplace = True)
        train_with_features = train_with_features.fillna(0)
        traindex_f = train_with_features.index
        print('train_with_features:', type(train_with_features), train_with_features.shape)

if os.path.isfile(test_store):
    print("loading train data from pickle file", test_store)
    with open(os.path.abspath(test_store), 'rb') as f:
        test_with_features = pickle.load(f, encoding='bytes')
        test_with_features.drop(['image'], axis=1, inplace = True)
        test_with_features = test_with_features.fillna(0)
        testdex_f = test_with_features.index
        feature_names = list(test_with_features.columns.values)
        print('test_with_features:', type(test_with_features), test_with_features.shape)

del f
gc.collect()



cat_features = list(
        test_with_features.dtypes[(test_with_features.dtypes != np.float16) &\
        (test_with_features.dtypes != np.float32) &\
        (test_with_features.dtypes != np.float64) &\
        (test_with_features.dtypes != np.uint64) &\
        (test_with_features.dtypes != np.int64)].index.values)
    
##train_with_features['Rand10'] = np.random.uniform(1, 10, train_with_features.shape[0])
##test_with_features['Rand10'] = np.random.uniform(1, 10, test_with_features.shape[0])

print( "Create multiple csr_matrix...")
start_time = time.time()

X_full = csr_matrix(train_with_features.loc[traindex].drop(['deal_probability'], axis=1))
print('X_full:', type(X_full), X_full.shape)
y_full = train_with_features.loc[traindex]['deal_probability']
print('y_full:', type(y_full), y_full.shape)

X_test_full = csr_matrix(test_with_features.loc[sub.index])
print('X_test_full:', type(X_test_full), X_test_full.shape)


validation_idx = holdout_index.index
train_idx = holdout_train_index.index

X_train = csr_matrix(train_with_features.loc[train_idx].drop(['deal_probability'], axis=1))
print('X_train:', type(X_train), X_train.shape)
y_train = train_with_features.loc[train_idx]['deal_probability']
print('y_train:', type(y_train), y_train.shape)


X_valid = csr_matrix(train_with_features.loc[validation_idx].drop(['deal_probability'], axis=1))
print('X_valid:', type(X_valid), X_valid.shape)
y_valid = train_with_features.loc[validation_idx]['deal_probability']
print('y_valid:', type(y_valid), y_valid.shape)


print('done in {} seconds.'.format(time.time() - start_time))

#
##save_npz('../input/features_%d_X_full.npz'%(len(test_with_features.columns)), X_full)
##save_npz('../input/features_%d_X_test_full.npz'%(len(test_with_features.columns)), X_test_full)
##save_npz('../input/features_%d_X_train.npz'%(len(test_with_features.columns)), X_train)
##save_npz('../input/features_%d_X_valid.npz'%(len(test_with_features.columns)), X_valid)
##y_csr_store = '../input/features_%d_y_for_csr.pkl'%(len(test_with_features.columns))
##print( "Saving y_for_csr data...")
##with open(os.path.abspath(y_csr_store), 'wb') as f:
##    pickle.dump((y_full, y_train, y_valid), f)
#
#
#

####################################
#
## ## Building a simple model
##
#feature_names = list(train_with_features.drop(['deal_probability'], axis=1).columns)
del train_with_features
del test_with_features, train_store, test_store

gc.collect()



print( "Create lgb.datasets...")
start_time = time.time()
dataset_train = lgb.Dataset(X_train, label=y_train,
                            feature_name = feature_names,
#                            categorical_feature = cat_features
                            )
dataset_valid = lgb.Dataset(X_valid, label=y_valid,
                            feature_name = feature_names,
#                            categorical_feature = cat_features,
                            reference = dataset_train)
print('done in {} seconds.'.format(time.time() - start_time))

#####################################################
#Model0
params = {'learning_rate': 0.04,
          'max_depth': 13,
          'boosting': 'gbdt',#dart, gbdt, goss
          'objective': 'xentropy',#'regression' #binary, xentropy
          'metric': ['rmse'],#,'auc'
          'is_training_metric': True,
          'seed': RANDOM_STATE,
          'num_leaves': 361,#512 #default 31
          'feature_fraction': 0.4,#default 1.0
#          'device': 'gpu',
          'bagging_fraction': 0.8,
          'bagging_freq': 5,
          'num_threads' : 4,
#          'max_cat_threshold' : 1000,
          }

print('number of features is ', len(feature_names))
evals_results = {}

start_time = time.time()
model0 = lgb.train(params,
                  dataset_train,
                  4000,
                  [
#                  dataset_train,
                   dataset_valid
                   ],
                  [#'train',
                          'valid'],
                  evals_result=evals_results,
                  verbose_eval=50,
                  early_stopping_rounds=200)
print('Model training without holdout done in {} seconds.'.format(time.time() - start_time))

## explain the model's predictions using SHAP values
#shap_values = shap.TreeExplainer(model0).shap_values(X_train)
#
## visualize the first prediction's explanation
#shap.force_plot(shap_values[0,:], X_train.iloc[0,:])
#
## summarize the effects of all the features
#shap.summary_plot(shap_values, X_train)

model0_store = "leonid_model{:d}_wo_holdout_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_{:.6g}.pkl".format(FILENO, len(feature_names),
           params['objective'], params['boosting'], params['max_depth'],
           params['num_leaves'], params['feature_fraction'],
           evals_results['valid']['rmse'][model0.best_iteration-1]
           )
print( "Saving model0 data...")
print(model0_store)
start_time = time.time()
with open(os.path.abspath(model0_store), 'wb') as f:
    pickle.dump((model0, evals_results), f, protocol = pickle.HIGHEST_PROTOCOL)
print('Done in {} seconds.'.format(time.time() - start_time))

# feature importances
model0_fi = pd.concat([pd.Series(model0.feature_name(), name = 'feature'),
        pd.Series(model0.feature_importance(importance_type='gain'), name = 'gain'),
        pd.Series(model0.feature_importance(), name = 'importance')
        ], axis=1).sort_values(by = ['gain'], ascending = False)

print('Save feature importances...')
start_time = time.time()
model0_fi.to_csv("model{:d}_fulltrain_{:d}_features_{:.6g}_{}_{}_md{:d}_nl{:d}_ff{:.1g}_fi_gain.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model0.best_iteration-1],
           params['objective'], params['boosting'], params['max_depth'],
           params['num_leaves'], params['feature_fraction']),
            index=False)
print('Done in {} seconds.'.format(time.time() - start_time))

#model0_store = "leonid_model21_wo_holdout_fulltrain_606_features_regression_gbdt_md13_nl512_ff0.4.pkl"
#if os.path.isfile(model0_store):
#    print("loading model w/o holdout data from pickle file", model0_store)
#    with open(os.path.abspath(model0_store), 'rb') as f:
#        model0, evals_results = pickle.load(f, encoding='bytes')
#feature_names = list(model0.feature_name())

print("model0.best_iteration: ", model0.best_iteration, "\n",
      'rmse (validation):',#train,
#      "{:.6g}".format(evals_results['train']['rmse'][model0.best_iteration-1]),
#      "{:.6g}".format(rmse(model0.predict(X_train), y_train)),
      "{:.6g}".format(evals_results['valid']['rmse'][model0.best_iteration-1]))



print('Predict holdout_v2 with model w/o holdout...')
pred_holdout = model0.predict(X_valid)
print('metric for holdout_v2 predict: rmse {:.6g}'.format(rmse(pred_holdout,
      y_valid)))
sub_holdout = pd.concat([pd.Series(y_valid.index.values, name = 'item_id'),
             pd.Series(np.clip(pred_holdout, 0, 1), name = 'deal_probability')],
             axis=1, verify_integrity=True)
#clipping is necessary.

sub_holdout.to_csv("../submit/leonid_model{:d}_predict_holdout_v2_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model0.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")

print('Predict train with model w/o holdout...')
pred_train = model0.predict(X_full)
print('metric for train_orig predict: rmse {:.6g}'.format(rmse(pred_train,
      y_full)))
sub_train = pd.concat([pd.Series(y_full.index.values, name = 'item_id'),
             pd.Series(np.clip(pred_train, 0, 1), name = 'deal_probability')],
        axis=1, verify_integrity=True)
#clipping is necessary.
sub_train.to_csv("../submit/leonid_model{:d}_predict_train_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model0.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")


print('Predict test with model w/o holdout...')
pred = model0.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv("../submit/leonid_model{:d}_holdout_predict_test_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model0.best_iteration-1],
           params['objective'], params['boosting']),
            index=True, float_format="%.5g")

print('Predict test with model w/o holdout V2...')
#pred = model0.predict(X_test_full)
#clipping is necessary.
sub_v2 = pd.concat([pd.Series(sub.index.values, name = 'item_id'),
             pd.Series(np.clip(pred, 0, 1), name = 'deal_probability')],
        axis=1, verify_integrity=True)
sub_v2.to_csv("../submit/leonid_model{:d}_holdout_predict_test_fulltrain_{:d}_features_{:.6g}_{}_{}_v2.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model0.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")


##############
print("Start learning model on full train data...")
start_time = time.time()
model0_f = lgb.train(params,
                  lgb.Dataset(X_full, label=y_full,
                              feature_name = feature_names),
                  model0.best_iteration,
#                  3200,
#                  init_model = model0,
                  verbose_eval=50)
print('Model training done in {} seconds.'.format(time.time() - start_time))

model0_f_store = "leonid_model{:d}_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_{:.6g}.pkl".format(FILENO, len(feature_names),
           params['objective'], params['boosting'], params['max_depth'],
           params['num_leaves'], params['feature_fraction'],
           evals_results['valid']['rmse'][model0.best_iteration-1])
print( "Saving full model data...")
start_time = time.time()
with open(os.path.abspath(model0_f_store), 'wb') as f:
    pickle.dump(model0_f, f, protocol = pickle.HIGHEST_PROTOCOL)
print('Done in {} seconds.'.format(time.time() - start_time))


#model0_f_store = "leonid_model21_fulltrain_606_features_regression_gbdt_md13_nl512_ff0.4.pkl"
#if os.path.isfile(model0_f_store):
#    print("loading fully trained model data from pickle file", model0_f_store)
#    with open(os.path.abspath(model0_f_store), 'rb') as f:
#        model0_f = pickle.load(f, encoding='bytes')



print('Predict test with full model...')
pred = model0_f.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv("../submit/leonid_model{:d}_predict_test_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model0.best_iteration-1],
           params['objective'], params['boosting']),
            index=True, float_format="%.5g")



