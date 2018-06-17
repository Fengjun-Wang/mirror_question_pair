# -*- coding: utf-8 -*-
import os
import gc
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def generate_feature_importance(gbm):
    fi = gbm.feature_importance()
    fn = gbm.feature_name()
    fin = zip(fi,fn)
    fin.sort(key=lambda x:x[0],reverse=True)
    res = pd.DataFrame(fin, columns=["Importance", "Feature"])
    res.to_csv("Feature_Importance.csv", index=False)


FEATURE_FOLDER = "../../inters/final_features"
FEATURE_TYPE = 1
train_file = "train_type%s.csv"%(FEATURE_TYPE)
test_file = "test_type%s.csv"%(FEATURE_TYPE)

df_train = pd.read_csv(os.path.join(FEATURE_FOLDER,train_file))
feature_names = df_train.columns.values.tolist()[:-1]

X_all = df_train.iloc[:,:-1].values
y_all = df_train["label"].values
del df_train
gc.collect()


X_train,X_val,y_train,y_val = train_test_split(X_all,y_all,test_size=0.25)
del X_all
del y_all
gc.collect()
print "number of train:",X_train.shape[0]
print "number of val:",X_val.shape[0]

lgb_train = lgb.Dataset(X_train, y_train,feature_name=feature_names)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,feature_name=feature_names)
del X_train
del y_train
del X_val
del y_val
gc.collect()
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 127,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 20,
    "device":"gpu",
    "max_bin":255
}
print('Start training...')
eva_res = {}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_eval], # eval training data
                valid_names = ["val"],
                feature_name=feature_names,
                early_stopping_rounds=50,
                learning_rates=lambda iter: 0.15 * (0.99 ** iter),
                evals_result = eva_res)
print('Save model...')
# save model to file
gbm.save_model('model.txt',num_iteration=gbm.best_iteration)

print("predicting")
df_test = pd.read_csv(os.path.join(FEATURE_FOLDER,test_file))
y_pred = gbm.predict(df_test.values, num_iteration=gbm.best_iteration)
y_pred = pd.DataFrame({"y_pre":y_pred})
y_pred.to_csv("submission.csv",index=False)
pickle.dump(eva_res,open("eva_res.pkl","wb"))

print "generating feature importance.."
generate_feature_importance(gbm)

