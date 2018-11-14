import numpy as np
import pandas as pd

RANDOM_STATE = 1542
np.random.seed(RANDOM_STATE)

# Thanks You Guillaume Martin for the Awesome Memory Optimizer!
# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if ((col_type != object) & (col_type != '<M8[ns]')):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


holdout_index = pd.read_csv('../data/holdout__index_itemid_v2.csv', index_col = 'item_id')

target = pd.read_csv('../input/train.csv', usecols = ['deal_probability','item_id'], index_col = 'item_id')
target = target.loc[holdout_index.index]

valid =  pd.read_csv('../submit/leonid_model21_predict_holdout_v2_fulltrain_606_features_0.215894_regression_gbdt.csv', index_col = 'item_id')

valid =  pd.read_csv('../submit/submit/leonid_model15_predict_holdout_v2_cleaned102train_214_features_0.215762_xentropy_gbdt_on_holdout_v2_0.186978.csv', index_col = 'item_id')

valid =  pd.read_csv('../submit/submit/leonid_model16_predict_holdout_v2_cleaned102train_214_features_0.215437_xentropy_gbdt_on_holdout_v2_0.189669.csv', index_col = 'item_id')

valid =  pd.read_csv('../submit/leonid_model22_predict_holdout_v2_fulltrain_587_features_0.216416_regression_gbdt.csv', index_col = 'item_id')

valid =  pd.read_csv('../submit/leonid_model23_predict_holdout_v2_fulltrain_513_features_0.217298_regression_gbdt.csv', index_col = 'item_id')

print(rmse(valid, target))
#deal_probability    0.296018
#dtype: float64

valid_red = reduce_mem_usage(valid)
target_red = reduce_mem_usage(target)

print(rmse(valid_red, target_red))
#deal_probability    0.296143
#dtype: float16

filename = 'leonid_model21_xentropy_gbdt_predict_holdout_0.21561_float_format_{}.csv'
float_f = ['not_set', '9g', '8g', '7g', '6g', '5g']
for i in float_f:
    print ('float format {} gives rmse:'.format(i))
    print(rmse(pd.read_csv(filename.format(i), index_col = 'item_id'),
               target))

valid_n =

leonid_model21_xentropy_gbdt_predict_holdout_0.21561_float_format_9g.csv
leonid_model21_xentropy_gbdt_predict_holdout_0.21561_float_format_8g.csv
leonid_model21_xentropy_gbdt_predict_holdout_0.21561_float_format_7g.csv
leonid_model21_xentropy_gbdt_predict_holdout_0.21561_float_format_6g.csv
leonid_model21_xentropy_gbdt_predict_holdout_0.21561_float_format_5g.csv
