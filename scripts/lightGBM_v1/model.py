# -*- coding: utf-8 -*-
import os
import gc
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

FEATURE_FOLDER = "../../inters/final_features"
FEATURE_TYPE = 1
train_file = "train_type%s.csv"%(FEATURE_TYPE)
test_file = "test_type%s.csv"%(FEATURE_TYPE)

df_train = pd.read_csv(os.path.join(FEATURE_FOLDER,train_file))
df_test = pd.read_csv(os.path.join(FEATURE_FOLDER,test_file))
feature_names = df_test.columns.values.tolist()

X_all = df_train.iloc[:,:-1].values
y_all = df_train["label"].values


X_train,X_val,y_train,y_val = train_test_split(X_all,y_all,test_size=0.25)
print "number of train:",X_train.shape[0]
print "number of val:",X_val.shape[0]

lgb_train = lgb.Dataset(X_train, y_train,feature_name=feature_names)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,feature_name=feature_names)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss',"binary_error"},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 7,
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
                early_stopping_rounds=30,
                evals_result = eva_res)
print('Save model...')
# save model to file
gbm.save_model('model.txt',num_iteration=gbm.best_iteration)

print("predicting")
y_pred = gbm.predict(df_test.values, num_iteration=gbm.best_iteration)
y_pred = pd.DataFrame({"y_pre":y_pred})
y_pred.to_csv("submission.csv",index=False)
pickle.dump(eva_res,open("eva_res.pkl","wb"))