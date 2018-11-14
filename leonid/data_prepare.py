# coding: utf-8
import os
import numpy as np
import pandas as pd

import time
from scipy import stats
from scipy.sparse import hstack, csr_matrix, load_npz, save_npz
from sklearn.model_selection import train_test_split
from collections import Counter
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
# from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
stop = set(stopwords.words('russian'))
import lightgbm as lgb
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import KFold
import gc
# try:
#    from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
#    from imblearn.over_sampling import RandomOverSampler
# except ImportError:
#    !pip install imblearn
#    from imblearn.under_sampling import RandomUnderSampler#, CondensedNearestNeighbour
#    from imblearn.over_sampling import RandomOverSampler

#import joblib
#from joblib import *
#sys.path.append("../scripts/") # Adds higher directory to python modules path.
#import extract_text_features
try:
    import cPickle as pickle
except:
    import pickle


pd.set_option('max_columns', 60)


RANDOM_STATE = 1542
np.random.seed(RANDOM_STATE)


FILENO = 23

sub = pd.read_csv('../input/sample_submission.csv', index_col = 'item_id')
holdout_index = pd.read_csv('../data/holdout__index_itemid_v2.csv', index_col = 'item_id')
holdout_train_index = pd.read_csv('../data/holdout_train__index_itemid_v2.csv', index_col = 'item_id')#, index_col = 'index'
holdout_train_folds_all_index = pd.read_csv('../data/holdout_train_folds_all__index_itemid_v2.csv')
train_folds_all_index = pd.read_csv('../data/train_folds_all__index_itemid_v2.csv', index_col = 'item_id')
#####################

train = pd.read_csv('../input/train.csv', index_col = 'item_id')
traindex = train.index
test = pd.read_csv('../input/test.csv', index_col = 'item_id')
testdex = test.index
##periods_train = pd.read_csv('../input/periods_train.csv')
#city_region_unique = pd.read_csv('../input/avito_region_city_features.csv')
#exchange_rates = pd.read_csv('../input/exchange_rates.csv')
#region_macro = pd.read_csv('../input/region_macro.csv')
#city_population_wiki = pd.read_csv('../input/city_population_wiki_v3.csv')
#aggregated_features = pd.read_csv('../input/aggregated_features.csv')
#
#################################
#print("Load text features")
#text_features_train_pkl = '../input/text_features_train.pkl'
#text_features_test_pkl = '../input/text_features_test.pkl'
#if os.path.isfile(text_features_train_pkl):
#    print("loading train data from pickle file", text_features_train_pkl)
#    with open(os.path.abspath(text_features_train_pkl), 'rb') as f:
#        text_features_train = pickle.load(f, encoding='bytes')
#if os.path.isfile(text_features_test_pkl):
#    print("loading test data from pickle file", text_features_test_pkl)
#    with open(os.path.abspath(text_features_test_pkl), 'rb') as f:
#        text_features_test = pickle.load(f, encoding='bytes')
#text_features_train.index = train.index
#text_features_test.index = test.index
#print('text_features_train:', type(text_features_train), text_features_train.shape)
#print('text_features_test:', type(text_features_test), text_features_test.shape)
#########################
#
#
#train_imgf = pd.read_csv('../input/train_img_features_v1.csv', index_col = 'item_id')
#test_imgf = pd.read_csv('../input/test_img_features_v1.csv', index_col = 'item_id')
#
##train.drop(index=['3131473e84a9','75ebe6b373ec'], inplace = True)
##train = train[pd.to_datetime(train.activation_date) <= pd.to_datetime('2017-03-28')]
#traindex = train.index
#
#
#
#y = train.deal_probability.copy().clip(0.0, 1.0)
#train.drop('deal_probability', axis=1, inplace=True)
#print('Train shape: {} Rows, {} Columns'.format(*train.shape))
#print('Test shape: {} Rows, {} Columns'.format(*test.shape))
#
##del train_imgf, test_imgf, text_features_train, text_features_test
#
#print("Combine Train and Test")
#df = pd.concat([train, test],axis=0)
#dfindex = df.index
#del train, test
#gc.collect()
#print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
#
#print("Add some external data")
#print("before:", df.shape)
#df = pd.merge(left=df, right=city_region_unique, how="left", on=["region", "city"])
#df = pd.merge(left=df, right=exchange_rates, how="left", on=["activation_date"])
#df = pd.merge(left=df, right=region_macro, how="left", on=["region"])
#df = pd.merge(left=df, right=city_population_wiki, how="left", on=["city"])
#df = pd.merge(left=df, right=aggregated_features, how="left", on=["user_id"])
## pd.merge does not keep the index so restore it
#df.index = dfindex
#print("after :", df.shape)
#del city_region_unique, exchange_rates, region_macro, city_population_wiki, aggregated_features
#
##ef = extract_text_features.extract_features(df)
#
#
## Thanks You Guillaume Martin for the Awesome Memory Optimizer!
## https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        print(col, type(df[col]), df[col].shape)
        col_type = df[col].dtype

        if ((col_type != object) & (col_type != '<M8[ns]') & (col_type.name != 'category')):#
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

def p25(values):
    return np.percentile(values, q=25)

def p75(values):
    return np.percentile(values, q=75)

def do_count(dataset, group_cols, show_agg=False):
    agg_name='{}_count'.format('_'.join(group_cols))
    if show_agg:
        print( "\nCounting by ", group_cols ,  '... and saving in', agg_name )
        start_time = time.time()
    try:
        dataset[agg_name] = dataset.groupby(group_cols)[set(dataset.columns).difference(set(group_cols)).pop()].transform('size').astype(np.uint32)
    except:
        print( "Error while counting by ", group_cols ,  '... and saving in', agg_name )
    if show_agg:
        print('Done in {} seconds.'.format(time.time() - start_time))
    gc.collect()
    return dataset

def do_agg(dataset, group_cols, counted, transformation, show_agg=False ):
    try:
        transformation_name = transformation.__name__
    except AttributeError:
        transformation_name = transformation
    agg_name= '{}_{}_by_{}'.format((transformation_name), (counted), ('_'.join(group_cols)))
    if show_agg:
        print( "\nTransform ", transformation_name, " on ", counted, " by ", group_cols ,  '... and saving in', agg_name )
        start_time = time.time()
    try:
        dataset[agg_name] = dataset[group_cols+[counted]].groupby(group_cols)[counted].transform(transformation).fillna(0)
    except:
        print( "Error while transform ", transformation_name, " on ", counted, " by ", group_cols ,  '... and saving in', agg_name)
    if show_agg:
        print('Done in {} seconds.'.format(time.time() - start_time))
    gc.collect()
    return dataset

def do_some_stats(dataset, group_cols, counted, show_agg=False ):
    dataset = do_agg(dataset, group_cols, counted, 'mean', show_agg)
    dataset = do_agg(dataset, group_cols, counted, 'median', show_agg)
    dataset = do_agg(dataset, group_cols, counted, 'max', show_agg)
    dataset = do_agg(dataset, group_cols, counted, 'min', show_agg)
    dataset = do_agg(dataset, group_cols, counted, 'var', show_agg)
    dataset = do_agg(dataset, group_cols, counted, 'std', show_agg)

    gc.collect()
    return dataset


def do_quantiles_advanced(dataset, group_cols, counted, show_agg=False ):
    agg_name_end = '{}_by_{}'.format((counted), ('_'.join(group_cols)))
    if show_agg:
        print( "\nTransform ", transformation, " on ", counted, " by ", group_cols ,  '... and saving in', agg_name )
        start_time = time.time()
    try:
        dataset['quantile_010_'+agg_name_end] = dataset[group_cols+[counted]].groupby(group_cols)[counted].quantile(0.1).fillna(0)
        dataset['quantile_025_'+agg_name_end] = dataset[group_cols+[counted]].groupby(group_cols)[counted].quantile(0.25).fillna(0)
        dataset['quantile_075_'+agg_name_end] = dataset[group_cols+[counted]].groupby(group_cols)[counted].quantile(0.75).fillna(0)
        dataset['quantile_090_'+agg_name_end] = dataset[group_cols+[counted]].groupby(group_cols)[counted].quantile(0.9).fillna(0)
#        dataset[agg_name] = dataset[group_cols+[counted]].groupby(group_cols)[counted].transform(transformation).fillna(0)
    except:
        print( "Error while quantiles_advanced ", " on ", counted, " by ", group_cols)
    if show_agg:
        print('Done in {} seconds.'.format(time.time() - start_time))
    gc.collect()
    return dataset

def text_feature_process (dataset):
    dataset['description'] = dataset['description'].apply(lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' '))

#    dataset['symbol1_count'] = dataset['description'].str.count('↓')
    dataset['symbol2_count'] = dataset['description'].str.count('\*')
#    dataset['symbol3_count'] = dataset['description'].str.count('✔')
#    dataset['symbol4_count'] = dataset['description'].str.count('❀')
    dataset['symbol5_count'] = dataset['description'].str.count('➚')
#    dataset['symbol6_count'] = dataset['description'].str.count('ஜ')
    dataset['symbol7_count'] = dataset['description'].str.count('.')
    dataset['symbol8_count'] = dataset['description'].str.count('!')
#    dataset['symbol9_count'] = dataset['description'].str.count('\?')
    dataset['symbol10_count'] = dataset['description'].str.count('  ')
    dataset['symbol11_count'] = dataset['description'].str.count('-')
    dataset['symbol12_count'] = dataset['description'].str.count(',')

    dataset['params'] = dataset['param_1'].fillna('') + ' ' + dataset['param_2'].fillna('') + ' ' + dataset['param_3'].fillna('')
#    dataset['params'] = dataset['params'].str.strip()

#     param_1, param_2, param_3, title, description
#    '_'.join(group_cols)

    dataset['len_title'] = dataset['title'].apply(lambda x: len(x))
#    dataset['words_title'] = dataset['title'].apply(lambda x: len(x.split()))
    dataset['len_description'] = dataset['description'].apply(lambda x: len(x))
#    dataset['words_description'] = dataset['description'].apply(lambda x: len(x.split()))
    dataset['len_params'] = dataset['params'].apply(lambda x: len(x))
#    dataset['words_params'] = dataset['params'].apply(lambda x: len(x.split()))

    textfeats = ['title', 'description']
    catfeats = ['category_name', 'parent_category_name', 'region',
                 'city', 'params']

    for cols in textfeats:
#        dataset[cols] = dataset[cols].astype(str)
        dataset[cols] = dataset[cols].astype(str).fillna('missing') # FILL NA
        dataset[cols] = dataset[cols].str.replace("[^[:alpha:]]", " ")
        dataset[cols] = dataset[cols].str.replace("\\s+", " ")
        dataset[cols] = dataset[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
#        dataset[cols + '_num_words'] = dataset[cols].apply(lambda comment: len(comment.split())) # Count number of Words
#        dataset[cols + '_num_unique_words'] = dataset[cols].apply(lambda comment: len(set(w for w in comment.split())))
#        dataset[cols + '_words_vs_unique'] = dataset[cols+'_num_unique_words'] / dataset[cols+'_num_words'] * 100 # Count Unique Words

    for cols in catfeats:
#        dataset[cols] = dataset[cols].astype(str)
        dataset[cols] = dataset[cols].astype(str).fillna('missing') # FILL NA
        dataset[cols] = dataset[cols].str.replace("[^[:alpha:]]", " ")
        dataset[cols] = dataset[cols].str.replace("\\s+", " ")
        dataset[cols] = dataset[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently


    dataset['text'] =  dataset['category_name'] + ' ' + dataset['parent_category_name'] + ' ' + dataset['region'] + ' ' + dataset['city'] + ' ' + dataset['params'] + ' ' + dataset['title'] + ' ' + dataset['description']

    return dataset

def data_process (dataset):
    dataset['activation_date'] = pd.to_datetime(dataset['activation_date'])
    dataset['weekday'] = dataset['activation_date'].dt.weekday

    dataset['has_image'] = 1
    dataset.loc[dataset['image'].isnull(), 'has_image'] = 0
    dataset['has_image'] = dataset['has_image'].astype('uint8')


    dataset['price'] = dataset.groupby(['city', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
    dataset['price'] = dataset.groupby(['region', 'category_name'])['price'].apply(lambda x: x.fillna(x.median()))
    dataset['price'] = dataset.groupby(['category_name'])['price'].apply(lambda x: x.fillna(x.median()))
    dataset['price_ln1p'] = np.log1p(dataset['price'].fillna(0))
    dataset['price_boxcox'] = stats.boxcox(dataset.price + 1)[0]
    dataset['price_usd'] = (dataset['price'] / dataset['USD_in_RUR']).round(2).fillna(0).astype('float32')
#    dataset['price'] = stats.boxcox(dataset.price + 1)[0]

    dataset = do_count(dataset, ['activation_date'])
    dataset = do_count(dataset, ['city', 'parent_category_name', 'category_name', 'params', 'user_type'])


    dataset = do_some_stats(dataset, ['region'], 'price')

    dataset = do_some_stats(dataset, ['city'], 'price')

    dataset = do_agg(dataset, ['parent_category_name'], 'price', 'mean')
    dataset = do_agg(dataset, ['parent_category_name'], 'price', 'max')

    dataset = do_agg(dataset, ['category_name'], 'price', 'max')
    dataset = do_agg(dataset, ['category_name'], 'price', 'median')
    dataset = do_agg(dataset, ['category_name'], 'price', 'mean')
    dataset = do_agg(dataset, ['category_name'], 'price', 'var')
    dataset = do_agg(dataset, ['category_name'], 'price', 'std')

    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name'], 'price')

    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name'], 'image_top_1')
    dataset = do_agg(dataset, ['city', 'parent_category_name', 'category_name'], 'image_top_1', stats.skew)



#    dataset['price'] = stats.boxcox(dataset.price + 1)[0]



    dataset = do_some_stats(dataset, ['user_id'], 'len_title')
    dataset = do_some_stats(dataset, ['user_id'], 'len_description')
    dataset = do_some_stats(dataset, ['user_id'], 'len_params')
    dataset = do_some_stats(dataset, ['user_id'], 'image_top_1')
    dataset = do_agg(dataset, ['user_id'], 'has_image', 'mean')
    dataset = do_agg(dataset, ['user_id'], 'has_image', 'var')
    dataset = do_some_stats(dataset, ['user_id'], 'price')
    dataset = do_some_stats(dataset, ['user_id'], 'price_ln1p')
    dataset = do_some_stats(dataset, ['user_id'], 'price_boxcox')
    dataset = do_some_stats(dataset, ['user_id'], 'price_usd')
    dataset = do_agg(dataset, ['user_id'], 'price_usd', stats.skew)

    dataset = do_agg(dataset, ['user_id'], 'item_seq_number', 'min')

    dataset = do_agg(dataset, ['user_type'], 'len_description', 'median')
    dataset = do_agg(dataset, ['user_type'], 'len_params', 'mean')
    dataset = do_agg(dataset, ['user_type'], 'len_params', 'std')
    dataset = do_agg(dataset, ['user_type'], 'image_top_1', 'median')
    dataset = do_agg(dataset, ['user_type'], 'image_top_1', 'mean')
    dataset = do_agg(dataset, ['user_type'], 'image_top_1', 'var')
    dataset = do_agg(dataset, ['user_type'], 'image_top_1', 'std')
    dataset = do_agg(dataset, ['user_type'], 'has_image', 'mean')
    dataset = do_agg(dataset, ['user_type'], 'has_image', 'var')
    dataset = do_agg(dataset, ['user_type'], 'has_image', 'std')

    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'price')
    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'len_title')
    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'len_description')
    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'image_top_1')
    dataset = do_agg(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'has_image', 'mean')
    dataset = do_agg(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'has_image', 'var')
    dataset = do_agg(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'has_image', 'std')
    dataset = do_agg(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'has_image', 'median')
    dataset = do_agg(dataset, ['city', 'parent_category_name', 'category_name', 'params'], 'has_image', 'min')

    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name', 'params', 'user_type'], 'price')
    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name', 'params', 'user_type'], 'len_title')
    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name', 'params', 'user_type'], 'len_description')
    dataset = do_some_stats(dataset, ['city', 'parent_category_name', 'category_name', 'params', 'user_type'], 'image_top_1')
    return dataset


print('Start text_process')
start_time = time.time()
df = text_feature_process(df)
print('text_process Done in {} seconds.'.format(time.time() - start_time))
gc.collect()

print('Start data_process')
start_time = time.time()
#train = data_process(train)
#test = data_process(test)
df = data_process(df)
print('data_process Done in {} seconds.'.format(time.time() - start_time))
gc.collect()

#df2 = df.copy()
#df.info()
df = reduce_mem_usage(df)
print('This step All Data shape: {} Rows, {} Columns'.format(*df.shape))
##df.info()
##df.columns
#
##df['activation_date'].sort_values(ascending=True).unique()
##array(['2017-03-15', '2017-03-16', '2017-03-17', '2017-03-18',
##       '2017-03-19', '2017-03-20', '2017-03-21', '2017-03-22',
##       '2017-03-23', '2017-03-24', '2017-03-25', '2017-03-26',
##       '2017-03-27', '2017-03-28', '2017-04-12', '2017-04-13',
##       '2017-04-14', '2017-04-15', '2017-04-16', '2017-04-17',
##       '2017-04-18', '2017-04-19', '2017-04-20'], dtype=object)
#
#print('Splitting back')
#train = pd.concat([df[:traindex.shape[0]],
#                   pd.Series(y, name = 'deal_probability')], axis=1)
#test = df[traindex.shape[0]:]
#print('Train shape: {} Rows, {} Columns'.format(*train.shape))
#print('Test shape: {} Rows, {} Columns'.format(*test.shape))
#
#
#traindex = train.index
#testdex = test.index
#train.reset_index(drop=True, inplace=True)
#test.reset_index(drop=True, inplace=True)
##del df, y
#gc.collect()
#
##https://www.kaggle.com/c/avito-duplicate-ads-detection/discussion/22190#126772
##Simple features
##category id
##number of images
##price difference
##Simple text features
##number of Russian and English characters in the title and description and number of digits and non alphanumeric characters; their ratio to all the text
##len of title and description
##number of unique chars
##cosine and jaccard of 2-3-4 ngrams on the char level
##fuzzy string distances calculated with FuzzyWuzzy (https://github.com/seatgeek/fuzzywuzzy)
##Simple image features
##stats (min, mean, max, std, skew, kurtosis) of each channel (R, G, B) as well as the average of all 3 channels
##file size
##number of geometry matches
##number of exact matches (calculated by md5)
##Simple Geo features
##metro id, location id
##distance between two locations
##With these features I was able to beat the avito benchmark
##
##Other features that I included afterwards:
##
##Attributes
##number of key matches, number of value matches, number of key-value pair matches
##number of fields that both ads didn't fill
##similarity of pairs in the tf-idf space, also svd of this space
##Text features
##jaccard and cosine only on digits and English tokens
##if any of the ads have english chars in a russian word (some of the characters look the same, but have different codes)
##tf, tf-idf and bm25 of title, description and all text
##svd of the above
##tf only on words that both ads have in common (in title, desc, all text), tf on words that only one of the ad has, svd of them
##distances and similarities in word2vec and glove spaces
##word2vec and glove similarity only on nouns
##some variation of word's mover distance for both w2v and glove
##how similar misspellings are in the ads
##I also tried to extract contact information (phones, emails, etc) but it didn't help much
##
##Image features
##image hashes from the imagehash library and from the forums
##phash from imagemagick computed on all individual channels
##diffirent similarities and distances of image histograms
##centroids and image moment invariants computed with imagemagick
##structural similarity of images
##SIFT and keypoint matching
##Geo features
##city, region and zip code extracted from geolocation (same sity, region, etc)
##dbscan and kmeans clusters of geolocations
##Meta features
##I run PCA and SVM on all the feature groups and used them as meta features
##Ensembling
##I computed too many features and it was not possible to fit them all into RAM (I have a 32gb/8cores machine) so I started ensembling pretty early. I randomly picked up 100-150 features and run XGBs or ETs on them.
##
##I mostly trained ETs because I could do ~10 of them per day, while training XGB took 2-3 days.
#
#
## ## Categorical features
##
## I'll use target encoding to deal with categorical features.
#
#
#
#def target_encode(trn_series=None,
#                  tst_series=None,
#                  target=None,
#                  min_samples_leaf=1,
#                  smoothing=1,
#                  noise_level=0):
#    """
#
#    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
#    Smoothing is computed like in the following paper by Daniele Micci-Barreca
#    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
#    trn_series : training categorical feature as a pd.Series
#    tst_series : test categorical feature as a pd.Series
#    target : target data as a pd.Series
#    min_samples_leaf (int) : minimum samples to take category average into account
#    smoothing (int) : smoothing effect to balance categorical average vs prior
#    """
#    assert len(trn_series) == len(target)
#    assert trn_series.name == tst_series.name
#    temp = pd.concat([trn_series, target], axis=1)
#    # Compute target mean
#    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
#    # Compute smoothing
#    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
#    # Apply average function to all target data
#    prior = target.mean()
#    # The bigger the count the less full_avg is taken into account
#    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
#    averages.drop(["mean", "count"], axis=1, inplace=True)
#    # Apply averages to trn and tst series
#    ft_trn_series = pd.merge(
#        trn_series.to_frame(trn_series.name),
#        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
#        on=trn_series.name,
#        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
#    # pd.merge does not keep the index so restore it
#    ft_trn_series.index = trn_series.index
#    ft_tst_series = pd.merge(
#        tst_series.to_frame(tst_series.name),
#        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
#        on=tst_series.name,
#        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
#    # pd.merge does not keep the index so restore it
#    ft_tst_series.index = tst_series.index
#    return ft_trn_series, ft_tst_series
#
#
#
#
#train['parent_category_name'], test['parent_category_name'] = target_encode(train['parent_category_name'], test['parent_category_name'], train['deal_probability'])
#train['category_name'], test['category_name'] = target_encode(train['category_name'], test['category_name'], train['deal_probability'])
#train['region'], test['region'] = target_encode(train['region'], test['region'], train['deal_probability'])
#train['image_top_1'], test['image_top_1'] = target_encode(train['image_top_1'], test['image_top_1'], train['deal_probability'])
#train['city'], test['city'] = target_encode(train['city'], test['city'], train['deal_probability'])
#train['param_1'], test['param_1'] = target_encode(train['param_1'], test['param_1'], train['deal_probability'])
#train['param_2'], test['param_2'] = target_encode(train['param_2'], test['param_2'], train['deal_probability'])
#train['param_3'], test['param_3'] = target_encode(train['param_3'], test['param_3'], train['deal_probability'])
#
#
#
#
#train.index = traindex
#test.index = testdex
#
#
#
#train = pd.concat([train, train_imgf.drop(['image'], axis=1), text_features_train], axis=1, verify_integrity=True)#.reset_index()
#test = pd.concat([test, test_imgf.drop(['image'], axis=1), text_features_test], axis=1, verify_integrity=True)#.reset_index()
#del train_imgf, test_imgf, text_features_train, text_features_test
#gc.collect()
#
#print('Train shape: {} Rows, {} Columns'.format(*train.shape))
#print('Test shape: {} Rows, {} Columns'.format(*test.shape))
#
#
#
#
#
#print('Start vectorizing')
## Now let's start transforming texts. Titles have little number of unique words, so we can use default values for TfidfVectorizer (only add stopwords). I have to limit max_features due to memory constraints. I won't use descriptions and parameters due to kernel limits.
#
#########################
##start_time = time.time()
##vectorizer=TfidfVectorizer(stop_words=stop, max_features=200000, norm='l2', sublinear_tf=True)
##vectorizer.fit(train['title'])
##train_title = vectorizer.transform(train['title'])
##test_title = vectorizer.transform(test['title'])
##print('Done in {} seconds.'.format(time.time() - start_time))
##gc.collect()
##
##start_time = time.time()
##vectorizer_d=TfidfVectorizer(stop_words=stop, max_features=200000, norm='l2', sublinear_tf=True)
##vectorizer_d.fit(train['description'])
##train_description = vectorizer_d.transform(train['description'])
##test_description = vectorizer_d.transform(test['description'])
##print('Done in {} seconds.'.format(time.time() - start_time))
##gc.collect()
##
##start_time = time.time()
##vectorizer_p=TfidfVectorizer(stop_words=stop, max_features=200000, norm='l2', sublinear_tf=True)
##vectorizer_p.fit(train['params'])
##train_params = vectorizer_p.transform(train['params'])
##test_params = vectorizer_p.transform(test['params'])
##print('Done in {} seconds.'.format(time.time() - start_time))
##gc.collect()
##
##start_time = time.time()
##vectorizer_t=TfidfVectorizer(stop_words=stop, max_features=200000, norm='l2', sublinear_tf=True)
##vectorizer_t.fit(train['text'])
##train_text = vectorizer_t.transform(train['text'])
##test_text = vectorizer_t.transform(test['text'])
##print('Done text in {} seconds.'.format(time.time() - start_time))
##gc.collect()
#
###################
#
#
#
###################
##save_npz('../input/train_title_tfidf_sparse_1_1_cleaned102train200000_l2.npz', train_title)
##save_npz('../input/test_title_tfidf_sparse_1_1_cleaned102train200000_l2.npz', test_title)
##save_npz('../input/train_description_tfidf_sparse_1_1_cleaned102train200000_l2.npz', train_description)
##save_npz('../input/test_description_tfidf_sparse_1_1_cleaned102train200000_l2.npz', test_description)
##save_npz('../input/train_params_tfidf_sparse_1_1_cleaned102train200000_l2.npz', train_params)
##save_npz('../input/test_params_tfidf_sparse_1_1_cleaned102train200000_l2.npz', test_params)
##save_npz('../input/train_text_tfidf_sparse_1_1_cleaned102train200000_l2.npz', train_text)
##save_npz('../input/test_text_tfidf_sparse_1_1_cleaned102train200000_l2.npz', test_text)
####################
#
#
##start_time = time.time()
##train_title = load_npz('../input/train_title_tfidf_sparse_1_1_fulltrain200000_l2.npz')
##test_title = load_npz('../input/test_title_tfidf_sparse_1_1_fulltrain200000_l2.npz')
##train_description = load_npz('../input/train_description_tfidf_sparse_1_1_fulltrain200000_l2.npz')
##test_description = load_npz('../input/test_description_tfidf_sparse_1_1_fulltrain200000_l2.npz')
##train_params = load_npz('../input/train_params_tfidf_sparse_1_1_fulltrain200000_l2.npz')
##test_params = load_npz('../input/test_params_tfidf_sparse_1_1_fulltrain200000_l2.npz')
##train_text = load_npz('../input/train_text_tfidf_sparse_1_1_fulltrain200000_l2.npz')
##test_text = load_npz('../input/test_text_tfidf_sparse_1_1_fulltrain200000_l2.npz')
##########
##train_tfidf14 = load_npz('../input/train_tfidf_sparse_1_4_clean_data_v1.npz')
##test_tfidf14 = load_npz('../input/test_tfidf_sparse_1_4_clean_data_v1.npz')
#########
##print('Loaded vectorized results in {} seconds.'.format(time.time() - start_time))
#
#gc.collect()
#
#
#
#
##train.head()
#
#
#
#
##train.columns
#
#kf = KFold(n_splits=7, random_state=9914, shuffle=True)
#
## One of possible ideas is creating meta-features. It means that we use some features to build a model and use the predictions in another model. I'll use ridge regression to create a new feature based on tokenized title and then I'll combine it with other features.
#
#print('Start creating meta features')
#print('Train shape: {} Rows, {} Columns'.format(*train.shape))
#print('Test shape: {} Rows, {} Columns'.format(*test.shape))
#
##start_time = time.time()
##X_meta = np.zeros((train_title.shape[0], 1))
##X_test_meta = []
##for fold_i, (train_i, test_i) in enumerate(kf.split(train_title)):
##    print('meta fold: ', fold_i)
##    model = Ridge()
##    model.fit(train_title.tocsr()[train_i], train['deal_probability'][train_i])
##    X_meta[test_i, :] = model.predict(train_title.tocsr()[test_i]).reshape(-1, 1)
##    X_test_meta.append(model.predict(test_title))
##
##X_test_meta = np.stack(X_test_meta)
##X_test_meta_mean = np.mean(X_test_meta, axis = 0).reshape(-1, 1)
##
##X_meta_d = np.zeros((train_description.shape[0], 1))
##X_test_meta_d = []
##for fold_i, (train_i, test_i) in enumerate(kf.split(train_description)):
##    print('meta_d fold: ', fold_i)
##    model_d = Ridge()
##    model_d.fit(train_description.tocsr()[train_i], train['deal_probability'][train_i])
##    X_meta_d[test_i, :] = model_d.predict(train_description.tocsr()[test_i]).reshape(-1, 1)
##    X_test_meta_d.append(model_d.predict(test_description))
##
##X_test_meta_d = np.stack(X_test_meta_d)
##X_test_meta_d_mean = np.mean(X_test_meta_d, axis = 0).reshape(-1, 1)
##
##X_meta_p = np.zeros((train_params.shape[0], 1))
##X_test_meta_p = []
##for fold_i, (train_i, test_i) in enumerate(kf.split(train_params)):
##    print('meta_p fold: ', fold_i)
##    model_p = Ridge()
##    model_p.fit(train_params.tocsr()[train_i], train['deal_probability'][train_i])
##    X_meta_p[test_i, :] = model_p.predict(train_params.tocsr()[test_i]).reshape(-1, 1)
##    X_test_meta_p.append(model_p.predict(test_params))
##
##X_test_meta_p = np.stack(X_test_meta_p)
##X_test_meta_p_mean = np.mean(X_test_meta_p, axis = 0).reshape(-1, 1)
##
##X_meta_t = np.zeros((train_text.shape[0], 1))
##X_test_meta_t = []
##for fold_i, (train_i, test_i) in enumerate(kf.split(train_text)):
##    print('meta_t fold: ', fold_i)
##    model_t = Ridge()
##    model_t.fit(train_text.tocsr()[train_i], train['deal_probability'][train_i])
##    X_meta_t[test_i, :] = model_t.predict(train_text.tocsr()[test_i]).reshape(-1, 1)
##    X_test_meta_t.append(model_t.predict(test_text))
##
##X_test_meta_t = np.stack(X_test_meta_t)
##X_test_meta_t_mean = np.mean(X_test_meta_t, axis = 0).reshape(-1, 1)
#
#
##X_meta_tfidf14 = np.zeros((train_tfidf14.shape[0], 1))
##X_test_meta_tfidf14 = []
##for fold_i, (train_i, test_i) in enumerate(kf.split(train_tfidf14)):
##    print('meta_tfidf14 fold: ', fold_i)
##    model_tfidf14 = Ridge()
##    model_tfidf14.fit(train_tfidf14.tocsr()[train_i], train['deal_probability'][train_i])
##    X_meta_tfidf14[test_i, :] = model_t.predict(train_tfidf14.tocsr()[test_i]).reshape(-1, 1)
##    X_test_meta_tfidf14.append(model_t.predict(test_tfidf14))
##
##X_test_meta_tfidf14 = np.stack(X_test_meta_tfidf14)
##X_test_meta_tfidf14_mean = np.mean(X_test_meta_tfidf14, axis = 0).reshape(-1, 1)
#
#
##train_cleaned = train.drop(index=['3131473e84a9','75ebe6b373ec'])
##train_cleaned = train_cleaned[pd.to_datetime(train_cleaned.activation_date) <= pd.to_datetime('2017-03-28')]
##train_dropped = pd.concat([train.loc[['3131473e84a9','75ebe6b373ec']],
##               train[pd.to_datetime(train.activation_date) > pd.to_datetime('2017-03-28')]], axis=0)
##
##
##train_tfidf14.index = train_cleaned.index
##test_tfidf14.index = testdex
##X_meta_tfidf14 = np.zeros((train_tfidf14.shape[0], 1))
##X_meta_tfidf14_dropped = []#np.zeros((train_dropped.shape[0], 1))
##X_test_meta_tfidf14 = []
##for fold_num in range(0, 7):
##    validation_idx = train_tfidf14.index.intersection(train_folds_all_index[train_folds_all_index.fold_validation == fold_num].index)
##    train_idx = train_tfidf14.index.intersection(train_folds_all_index[train_folds_all_index.fold_validation != fold_num].index)
##    print('validation fold number: ', fold_num,
##          'validation size: ', len(validation_idx),
##          'train size: ', len(train_idx))
##    model_tfidf14 = Ridge()
##    model_tfidf14.fit(train_tfidf14[train_idx.factorize()[0]], train_cleaned['deal_probability'][train_idx])
##    X_meta_tfidf14[validation_idx.factorize()[0], :] = model_tfidf14.predict(train_tfidf14.tocsr()[validation_idx.factorize()[0]]).reshape(-1, 1)
##    X_meta_tfidf14_dropped.append(np.median(X_meta_tfidf14[validation_idx.factorize()[0], :]))
##    X_test_meta_tfidf14.append(model_tfidf14.predict(test_tfidf14))
##
##X_meta_tfidf14_full = pd.Series(np.zeros((train.shape[0],)), name = 'meta_feature_tfidf14')
##X_meta_tfidf14_full.index = traindex
##X_meta_tfidf14_full[train_cleaned.index] = X_meta_tfidf14.flatten()
##X_meta_tfidf14_full[train_dropped.index] = np.mean(X_meta_tfidf14_dropped)
##X_meta_tfidf14_full = pd.DataFrame(X_meta_tfidf14_full)
##
##X_test_meta_tfidf14 = np.stack(X_test_meta_tfidf14)
##X_test_meta_tfidf14_mean = np.mean(X_test_meta_tfidf14, axis = 0).reshape(-1, 1)
##
##X_meta_tfidf14_full.to_csv('../input/meta_features_tfidf14_train_arch.csv')
##meta_features_tfidf14_test_arch = pd.Series(X_test_meta_tfidf14_mean.flatten(), name = 'meta_feature_tfidf14')
##meta_features_tfidf14_test_arch.index = testdex
##meta_features_tfidf14_test_arch = pd.DataFrame(meta_features_tfidf14_test_arch)
##meta_features_tfidf14_test_arch.to_csv('../input/meta_features_tfidf14_test_arch.csv')
#
#
#print('Done in {} seconds.'.format(time.time() - start_time))
#
##'date', 'day',
#train.drop(['user_id', 'param_1', 'param_2', 'param_3', 'city_region'], axis=1, inplace=True)
#test.drop(['user_id', 'param_1', 'param_2', 'param_3', 'city_region'], axis=1, inplace=True)
##train.drop(['city_region'], axis=1, inplace=True)
##test.drop(['city_region'], axis=1, inplace=True)
#
#train.drop(['title', 'params', 'description', 'user_type',  'image', 'text'], axis=1, inplace=True)#, 'item_id',
#test.drop(['title', 'params', 'description', 'user_type', 'image', 'text'], axis=1, inplace=True)#, 'item_id',
#
#train.drop(['activation_date'], axis=1, inplace=True)
#test.drop(['activation_date'], axis=1, inplace=True)
#gc.collect()
#
#try:
#    del train_title, test_title
#    del train_description, test_description
#    del train_params, test_params
#    del train_text, test_text
##    del train_tfidf14, test_tfidf14
#    del fold_i, train_i, test_i
##    del vectorizer, vectorizer_d, vectorizer_p, vectorizer_t
##    del model, model_d, model_p, model_t, model_tfidf14
#    print ('unneeded vars.deleted')
#except NameError:
#    print ('some unneeded vars already deleted')
#
#gc.collect()
#
#
##meta_features_train_arch = pd.concat([
##             pd.Series(X_meta.flatten(), name = 'meta_feature'),
##             pd.Series(X_meta_d.flatten(), name = 'meta_feature_d'),
##             pd.Series(X_meta_p.flatten(), name = 'meta_feature_p'),
##             pd.Series(X_meta_t.flatten(), name = 'meta_feature_t'),
##             pd.Series(X_meta_tfidf14.flatten(), name = 'meta_feature_tfidf14')
##             ], axis=1)
##meta_features_train_arch.index = traindex
##meta_features_train_arch.to_csv('../input/meta_features_cleaned102train_train_arch.csv')
##
##meta_features_test_arch = pd.concat([
##             pd.Series(X_test_meta_mean.flatten(), name = 'meta_feature'),
##             pd.Series(X_test_meta_d_mean.flatten(), name = 'meta_feature_d'),
##             pd.Series(X_test_meta_p_mean.flatten(), name = 'meta_feature_p'),
##             pd.Series(X_test_meta_t_mean.flatten(), name = 'meta_feature_t'),
##             pd.Series(X_test_meta_tfidf14_mean.flatten(), name = 'meta_feature_tfidf14')
##             ], axis=1)
##meta_features_test_arch.index = testdex
##meta_features_test_arch.to_csv('../input/meta_features_cleaned102train_test_arch.csv')
#
#meta_features_train_arch = pd.read_csv('../input/meta_features_train_arch.csv', index_col = 'item_id')
#meta_features_test_arch = pd.read_csv('../input/meta_features_test_arch.csv', index_col = 'item_id')
#
#
#train_with_features = pd.concat([train,
#                                 meta_features_train_arch], axis=1)
#
#test_with_features = pd.concat([test,
#                                 meta_features_test_arch], axis=1)
#
#
#dense_train_f = pd.read_csv('../input/dense_train.csv', index_col = 'item_id')
#dense_test_f = pd.read_csv('../input/dense_test.csv', index_col = 'item_id')
#dense_train_f = dense_train_f.drop(['user_id', 'region', 'city',
#        'parent_category_name', 'category_name',
#       'param_1', 'param_2', 'param_3', 'price', 'item_seq_number',
#       'user_type', 'image_top_1'], axis=1)
#dense_test_f = dense_test_f.drop(['user_id', 'region', 'city',
#        'parent_category_name', 'category_name',
#       'param_1', 'param_2', 'param_3', 'price', 'item_seq_number',
#       'user_type', 'image_top_1'], axis=1)
#
#fasttext_train_f = pd.read_csv('../input/fasttext_train.csv', index_col = 'item_id')
#fasttext_test_f = pd.read_csv('../input/fasttext_test.csv', index_col = 'item_id')
#
#dense_train_f = reduce_mem_usage(dense_train_f)
#dense_test_f = reduce_mem_usage(dense_test_f)
#fasttext_train_f = reduce_mem_usage(fasttext_train_f)
#fasttext_test_f = reduce_mem_usage(fasttext_test_f)
#
#train_with_features = pd.concat([train_with_features,
#                                 dense_train_f,
#                                 fasttext_train_f], axis=1)
#
#test_with_features = pd.concat([test_with_features,
#                                dense_test_f,
#                                fasttext_test_f], axis=1)
#
##train_store = './train_with_features_%d.pkl'%(len(test_with_features.columns))
##test_store = './test_with_features_%d.pkl'%(len(test_with_features.columns))
##print( "Saving training data...")
##with open(os.path.abspath(train_store), 'wb') as f:
##    pickle.dump(train_with_features, f)
##
##print( "Saving testing data...")
##with open(os.path.abspath(test_store), 'wb') as f:
##    pickle.dump(test_with_features, f)
#
##del dense_train_f, dense_test_f, fasttext_train_f, fasttext_test_f

train_store = './train_with_features_606.pkl'
test_store = './test_with_features_606.pkl'
if os.path.isfile(train_store):
    print("loading train data from pickle file", train_store)
    with open(os.path.abspath(train_store), 'rb') as f:
        train_with_features = pickle.load(f, encoding='bytes')
        traindex_f = train_with_features.index
        print('train_with_features:', type(train_with_features), train_with_features.shape)

if os.path.isfile(test_store):
    print("loading train data from pickle file", test_store)
    with open(os.path.abspath(test_store), 'rb') as f:
        test_with_features = pickle.load(f, encoding='bytes')
        testdex_f = test_with_features.index
        feature_names = list(test_with_features.columns.values)
        print('test_with_features:', type(test_with_features), test_with_features.shape)


del f

y_f = train_with_features.deal_probability.copy().clip(0.0, 1.0)
train_with_features.drop('deal_probability', axis=1, inplace=True)

print("Combine Train and Test")
df_f = pd.concat([train_with_features, test_with_features], axis=0)
df_f_index = df_f.index
#del train, test
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df_f.shape))

#temp = df_f['title_num_words'].iloc[:,0]
#df_f.drop(['title_num_words'], axis = 1, inplace = True)
#df_f['title_num_words'] = temp
#
#temp = df_f['title_num_unique_words'].iloc[:,0]
#df_f.drop(['title_num_unique_words'], axis = 1, inplace = True)
#df_f['title_num_unique_words'] = temp
#
#temp = df_f['title_words_vs_unique'].iloc[:,0]
#df_f.drop(['title_words_vs_unique'], axis = 1, inplace = True)
#df_f['title_words_vs_unique'] = temp
#
#temp = df_f['description_num_words'].iloc[:,0]
#df_f.drop(['description_num_words'], axis = 1, inplace = True)
#df_f['description_num_words'] = temp
#
#temp = df_f['description_num_unique_words'].iloc[:,0]
#df_f.drop(['description_num_unique_words'], axis = 1, inplace = True)
#df_f['description_num_unique_words'] = temp
#
#temp = df_f['description_words_vs_unique'].iloc[:,0]
#df_f.drop(['description_words_vs_unique'], axis = 1, inplace = True)
#df_f['description_words_vs_unique'] = temp


#df_f = reduce_mem_usage(df_f)
#Memory usage after optimization is: 2599.20 MB

#df_f = df_f[feature_names]

###df_f.loc[traindex_f]
###pd.concat([df_f.loc[traindex], y_f.loc[traindex]], axis=1)
###print('Splitting back')
###train = pd.concat([df[:traindex.shape[0]],
###                   pd.Series(y, name = 'deal_probability')], axis=1)
###test = df[traindex.shape[0]:]
###print('Train shape: {} Rows, {} Columns'.format(*train.shape))
###print('Test shape: {} Rows, {} Columns'.format(*test.shape))


train_store = './train_with_features_%d.pkl'%(len(feature_names))
test_store = './test_with_features_%d.pkl'%(len(feature_names))
print( "Saving training data...")
with open(os.path.abspath(train_store), 'wb') as f:
    pickle.dump(pd.concat([df_f.loc[traindex], y_f.loc[traindex]], axis=1),
                f, protocol = pickle.HIGHEST_PROTOCOL)

print( "Saving testing data...")
with open(os.path.abspath(test_store), 'wb') as f:
    pickle.dump(df_f.loc[sub.index], f, protocol = pickle.HIGHEST_PROTOCOL)


#df_f.describe()

some_fi_multi = pd.concat([pd.read_csv('model16_cleaned102train_214_features_0.215437_xentropy_gbdt_md9_nl128_ff0.9_fi_gain.csv'),
                           pd.read_csv('model15_cleaned102train_214_features_0.215762_xentropy_gbdt_md17_nl128_ff0.9_fi_gain.csv'),
                           pd.read_csv('model21_fulltrain_606_features_0.215894_regression_gbdt_md13_nl512_ff0.4_fi_gain.csv'),
                           pd.read_csv('model21_fulltrain_606_features_0.217033_xentropy_goss_md13_nl512_ff0.4_fi_gain.csv'),
                           pd.read_csv('model21_fulltrain_606_features_0.21561_xentropy_gbdt_md13_nl512_ff0.4_fi_gain.csv'),
                           pd.read_csv('model22_fulltrain_587_features_0.216416_regression_gbdt_md13_nl512_ff0.4_fi_gain.csv')
        ], axis = 0)
drop_fi2_multi = set(some_fi_multi[some_fi_multi.gain<1].feature)

df_f.drop(list(set(list(df_f.columns)).intersection(drop_fi2_multi)),
          axis = 1, inplace = True)



train_store_2 = './train_with_features_587.pkl'
test_store_2 = './test_with_features_587.pkl'
if os.path.isfile(train_store_2):
    print("loading train data from pickle file", train_store_2)
    with open(os.path.abspath(train_store_2), 'rb') as f:
        train_with_features_2 = pickle.load(f, encoding='bytes')
        traindex_f_2 = train_with_features_2.index
        print('train_with_features_2:', type(train_with_features_2), train_with_features_2.shape)

if os.path.isfile(test_store_2):
    print("loading train data from pickle file", test_store_2)
    with open(os.path.abspath(test_store_2), 'rb') as f:
        test_with_features_2 = pickle.load(f, encoding='bytes')
        testdex_f_2 = test_with_features_2.index
        print('test_with_features:', type(test_with_features_2), test_with_features_2.shape)

del f

y_f_2 = train_with_features_2.deal_probability.copy().clip(0.0, 1.0)
train_with_features_2.drop('deal_probability', axis=1, inplace=True)

print("Combine Train and Test")
df_f_2 = pd.concat([train_with_features_2, test_with_features_2], axis=0)
df_f_2_index = df_f_2.index
#del train, test
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df_f_2.shape))

df_f_2.drop(list(set(list(df_f_2.columns)).intersection(drop_fi2_multi)),
          axis = 1, inplace = True)

len(set(df_f_2.columns).intersection(set(df_f.columns)))

set(df_f.columns).difference(set(df_f_2.columns))
set(df_f_2.columns).difference(set(df_f.columns))

df_f_2.drop(['meta_feature', 'meta_feature_d', 'meta_feature_p',
             'meta_feature_t', 'meta_feature_tfidf14',
             'item_seq_number', 'nima_nasnet_mean', 'nima_nasnet_std',
             'Weekday', 'image_top_1'],
          axis = 1, inplace = True)

for col in df_f_2.columns:
    if (df_f_2[col] % 1 == 0).all():
        df_f_2[col] = df_f_2[col].astype(np.int64)
gc.collect()

df_f_2 = reduce_mem_usage(df_f_2)

#(df_f_2.n_user_items[0:11] % 1 == 0).all()


#
#'n_user_items', 'symbol2_count', 'symbol7_count', 'symbol8_count', 'symbol10_count', 'symbol11_count', 'symbol12_count', 'len_title', 'len_description', 'len_params', 'weekday', 'activation_date_count', 'city_parent_category_name_category_name_params_user_type_count','user_id_count', 'region_count', 'city_count', 'parent_category_name_count', 'category_name_count', 'user_type_count', 'image_top_1_count', 'param_1_count', 'param_2_count', 'param_3_count', 'title_count', 'description_starts_with_title',  'title_description_intersection', 'title_description_short_intersection', 'description_short_num_words', 'description_short_num_unique_words', 'description_num_words', 'description_num_unique_words',  'title_num_words', 'title_num_unique_words'
#
#
#user_id_count,region_count,city_count,parent_category_name_count,category_name_count,user_type_count,image_top_1_count,param_1_count,param_2_count,param_3_count,title_count,Weekday,title_encoded,description_num_words,description_num_unique_words,description_words_vs_unique,title_num_words,title_num_unique_words,title_words_vs_unique,description_short_num_words,description_short_num_unique_words,description_short_words_vs_unique,title_description_intersection,title_description_short_intersection,title_description_intersection_vs_unique,title_description_short_intersection_vs_unique,title_equals_description,description_starts_with_title
#df_f_2.title_num_unique_words.astype(np.uint8)

dense_types = {
        'user_id'              : 'uint32',
        'region'               : 'uint8',
        'city'                 : 'uint16',
        'parent_category_name' : 'uint8',
        'category_name'        : 'uint8',
        'param_1'              : 'uint16',
        'param_2'              : 'uint16',
        'param_3'              : 'uint16',
        'price'                : 'float16',
        'item_seq_number'      : 'uint32',
        'user_type'            : 'uint8',
        }

dense_cols = ['item_id', 'user_id', 'region', 'city',
        'parent_category_name', 'category_name',
       'param_1', 'param_2', 'param_3', 'price', 'item_seq_number',
       'user_type']
dense_train_f = pd.read_csv('../input/dense_train.csv', index_col = 'item_id',
                            usecols = dense_cols, dtype = dense_types)
dense_test_f = pd.read_csv('../input/dense_test.csv', index_col = 'item_id',
                            usecols = dense_cols, dtype = dense_types)

dense_f = pd.concat([dense_train_f, dense_test_f],
                    axis=0, verify_integrity=True)

dense_f.rename(columns = {"price": "price_dense"}, inplace = True)


nima_nasnet_features_train = pd.read_csv('../julia/nima_nasnet_features_train.csv', usecols = ['item_id', 'mean', 'std'], index_col = 'item_id')
nima_nasnet_features_test = pd.read_csv('../julia/nima_nasnet_features_test.csv', usecols = ['item_id', 'mean', 'std'], index_col = 'item_id')

nima_nasnet_features = pd.concat([nima_nasnet_features_train,
                                  nima_nasnet_features_test],
                    axis=0, verify_integrity=True)

nima_nasnet_features.rename(columns={"mean": "nima_nasnet_mean",
                                           "std": "nima_nasnet_std"},
                                    inplace = True)

#nima_nasnet_features.reset_index(inplace = True)

nima_nasnet = pd.DataFrame(pd.concat([train.image.fillna(0),
                                  test.image.fillna(0)],
                    axis=0, verify_integrity=True))
nima_nasnet_features = pd.merge(left=nima_nasnet, right=nima_nasnet_features, how="left", on=['item_id']).fillna(0)
nima_nasnet_features.index = nima_nasnet.index
nima_nasnet_features.drop(['image'], axis = 1, inplace = True)
#del nima_nasnet_features_train, nima_nasnet_features_test, nima_nasnet
gc.collect()


resnet101_features_train = pd.read_csv('../input/resnet_features_top5.csv')
resnet101_features_test = pd.read_csv('../input/resnet_features_top5_test.csv')

resnet101_features_train.rename(columns = {"Unnamed: 0": "image"}, inplace = True)
resnet101_features_test.rename(columns = {"Unnamed: 0": "image"}, inplace = True)

resnet101_features_train.set_index("image", inplace = True)
resnet101_features_test.set_index("image", inplace = True)

resnet101_features = pd.concat([resnet101_features_train,
                                resnet101_features_test],
                    axis=0, verify_integrity=True)

resnet101_features.reset_index(inplace = True)

resnet101 = pd.DataFrame(pd.concat([train.image.fillna(0),
                                  test.image.fillna(0)],
                    axis=0, verify_integrity=True))


resnet101_features = pd.merge(left=resnet101, right=resnet101_features, how="left", on=["image"]).fillna(1001)
resnet101_features.index = resnet101.index

resnet101_features = resnet101_features.astype(dtype = {"resnet_101_top_0": "uint16",
                                                        "resnet_101_top_1": "uint16",
                                                        "resnet_101_top_2": "uint16",
                                                        "resnet_101_top_3": "uint16",
                                                        "resnet_101_top_4": "uint16"                                      })

del dense_train_f, dense_test_f, drop_fi, drop_fi2, feature_names, model0, nima_nasnet_features_train, nima_nasnet_features_test, resnet101_features_test, resnet101_features_train, resnet101, some_fi, some_fi2, temp, y_f, y_f_2
gc.collect()

train_with_features_3 = pd.concat([dense_f.loc[traindex],
                                   df_f_2.loc[traindex],
                                   nima_nasnet_features.loc[traindex],
                                   resnet101_features.loc[traindex],
                                   train.loc[traindex][['image_top_1', 'deal_probability']]
                                   ], axis=1, verify_integrity=True)

test_with_features_3 = pd.concat([dense_f.loc[testdex],
                                   df_f_2.loc[testdex],
                                   nima_nasnet_features.loc[testdex],
                                   resnet101_features.loc[testdex],
                                   test.loc[testdex]['image_top_1']
                                   ], axis=1, verify_integrity=True)


train_store = './train_with_features_%d_no_meta.pkl'%(len(test_with_features_3.columns))
test_store = './test_with_features_%d_no_meta.pkl'%(len(test_with_features_3.columns))
print( "Saving training data...")
with open(os.path.abspath(train_store), 'wb') as f:
    pickle.dump(train_with_features_3,
                f, protocol = pickle.HIGHEST_PROTOCOL)

print( "Saving testing data...")
with open(os.path.abspath(test_store), 'wb') as f:
    pickle.dump(test_with_features_3, f, protocol = pickle.HIGHEST_PROTOCOL)

#temp = df_f['title_num_words'].iloc[:,0]
#df_f.drop(['title_num_words'], axis = 1, inplace = True)
#df_f['title_num_words'] = temp
#
#temp = df_f['title_num_unique_words'].iloc[:,0]
#df_f.drop(['title_num_unique_words'], axis = 1, inplace = True)
#df_f['title_num_unique_words'] = temp
#
#temp = df_f['title_words_vs_unique'].iloc[:,0]
#df_f.drop(['title_words_vs_unique'], axis = 1, inplace = True)
#df_f['title_words_vs_unique'] = temp
#
#temp = df_f['description_num_words'].iloc[:,0]
#df_f.drop(['description_num_words'], axis = 1, inplace = True)
#df_f['description_num_words'] = temp
#
#temp = df_f['description_num_unique_words'].iloc[:,0]
#df_f.drop(['description_num_unique_words'], axis = 1, inplace = True)
#df_f['description_num_unique_words'] = temp
#
#temp = df_f['description_words_vs_unique'].iloc[:,0]
#df_f.drop(['description_words_vs_unique'], axis = 1, inplace = True)
#df_f['description_words_vs_unique'] = temp




#train_with_features['Rand10'] = np.random.uniform(1, 10, train_with_features.shape[0])
#test_with_features['Rand10'] = np.random.uniform(1, 10, test_with_features.shape[0])

#fi_lower22 = ['description_num_russian_words', 'description_num_187_chars',
#'all_text_num_95_chars', 'description_num_8221_chars',
#'all_text_num_8211_chars', 'description_num_8595_chars',
#'all_text_num_178_chars', 'all_text_num_174_chars',
#'all_text_num_10004_chars', 'description_num_64_chars',
#'all_text_num_62_chars', 'all_text_num_215_chars', 'all_text_num_8226_chars',
#'all_text_num_183_chars', 'all_text_num_64_chars',
#'description_num_8230_chars', 'all_text_num_8901_chars',
#'description_num_8220_chars', 'title_num_russian_words',
#'all_text_num_other_chars', 'all_text_num_61_chars',
#'description_num_178_chars', 'all_text_num_176_chars',
#'all_text_num_8230_chars', 'all_text_num_9658_chars', 'title_num_62_chars',
#'title_num_other_chars', 'all_text_num_63_chars',
#'min_price_by_category_name', 'median_image_top_1_by_user_type',
#'title_num_174_chars', 'title_num_126_chars', 'title_num_95_chars',
#'has_image', 'title_num_38_chars', 'min_len_params_by_user_type',
#'title_num_64_chars', 'description_ratio_russian_words',
#'max_image_top_1_by_user_type', 'description_ratio_english_words',
#'title_num_63_chars', 'title_num_176_chars', 'title_num_61_chars',
#'min_has_image_by_user_type', 'title_num_37_chars',
#'max_has_image_by_city_parent_category_name_category_name_params',
#'title_num_35_chars', 'max_len_params_by_user_type',
#'all_text_ratio_english_words', 'var_has_image_by_user_type',
#'all_text_ratio_russian_words', 'max_has_image_by_user_id',
#'min_len_description_by_user_type', 'symbol5_count',
#'median_has_image_by_user_type', 'max_has_image_by_user_type',
#'min_image_top_1_by_user_type', 'max_len_description_by_user_type',
#'std_has_image_by_user_type', 'title_num_8211_chars', 'title_num_183_chars',
#'title_num_8381_chars', 'all_text_num_8221_chars', 'all_text_num_8220_chars',
#'all_text_num_8212_chars', 'all_text_num_2972_chars',
#'all_text_num_1769_chars', 'all_text_num_1758_chars', 'title_num_8226_chars',
#'title_num_8230_chars', 'title_num_8595_chars', 'title_ratio_other_chars',
#'title_num_8901_chars', 'title_num_9552_chars', 'title_num_9658_chars',
#'title_num_9742_chars', 'title_num_10004_chars', 'title_num_10048_chars',
#'description_num_10048_chars', 'title_num_65039_chars',
#'all_text_num_8595_chars', 'all_text_num_9552_chars',
#'all_text_num_9742_chars', 'all_text_num_10048_chars', 'title_num_187_chars',
#'title_num_215_chars', 'title_num_1758_chars', 'description_num_1758_chars',
#'title_ratio_english_words', 'title_ratio_russian_words',
#'title_num_1769_chars', 'mean_has_image_by_user_type',
#'title_num_2972_chars', 'description_num_2972_chars', 'min_price_by_region',
#'title_num_128077_chars', 'title_num_8212_chars', 'title_num_8220_chars',
#'title_num_8221_chars', 'all_text_num_128077_chars',
#'all_text_num_65039_chars', 'description_num_1769_chars']
#
#train_with_features = train_with_features.drop(fi_lower22, axis=1)
#test_with_features = test_with_features.drop(fi_lower22, axis=1)

#train_with_features = reduce_mem_usage(train_with_features)
#test_with_features = reduce_mem_usage(test_with_features)

print( "Create multiple csr_matrix...")
start_time = time.time()

X_full = csr_matrix(train_with_features.drop(['deal_probability'], axis=1))
print('X_full:', type(X_full), X_full.shape)
y_full = train_with_features['deal_probability']
print('y_full:', type(y_full), y_full.shape)

X_test_full = csr_matrix(test_with_features)
print('X_test_full:', type(X_test_full), X_test_full.shape)

#holdout_train_index.index
#holdout_index.index
X_full.index = traindex
validation_idx = holdout_index.index.intersection(train_with_features.index)
train_idx = holdout_train_index.index.intersection(train_with_features.index)

X_train = csr_matrix(train_with_features.loc[train_idx].drop(['deal_probability'], axis=1))
print('X_train:', type(X_train), X_train.shape)
y_train = train_with_features.loc[train_idx]['deal_probability']
print('y_train:', type(y_train), y_train.shape)
#y_train = train_with_features['deal_probability'].reset_index(drop=True)[holdout_train_index.index]

X_valid = csr_matrix(train_with_features.loc[validation_idx].drop(['deal_probability'], axis=1))
print('X_valid:', type(X_valid), X_valid.shape)
y_valid = train_with_features.loc[validation_idx]['deal_probability']
print('y_valid:', type(y_valid), y_valid.shape)


print('done in {} seconds.'.format(time.time() - start_time))
#y_valid = train['deal_probability'].reset_index(drop=True)[holdout_index.index]

#X_train, X_valid, y_train, y_valid = train_test_split(X_full, train['deal_probability'], test_size=0.20, random_state=RANDOM_STATE)

#################################
#train_orig = train.copy()
#train['deal'] = (train.deal_probability >= 0.01)

#X_train_mod = train.drop(['item_id', 'image'], axis=1)


#X_train_mod = pd.concat([train,
#             pd.Series(X_meta.flatten(), name = 'meta_feature'),
#             pd.Series(X_meta_d.flatten(), name = 'meta_feature_d'),
#             pd.Series(X_meta_p.flatten(), name = 'meta_feature_p'),
#             pd.Series(X_meta_t.flatten(), name = 'meta_feature_t'),
##             pd.Series(X_meta_tfidf14.flatten(), name = 'meta_feature_tfidf14')
#             ], axis=1)
##y_train_mod = train['deal']
#y_train_mod = (train.deal_probability >= 0.01)
##y_train = train.deal_probability
#
#print('Original dataset shape {}'.format(Counter(y_train_mod)))
#rus = RandomUnderSampler(random_state = RANDOM_STATE)
#X_res, y_res = rus.fit_sample(X_train_mod.fillna(0), y_train_mod)
#X_res = pd.DataFrame(X_res, columns = X_train_mod.columns)
##y_res = pd.Series(y_res, name = y_train_mod.name)
#print('Resampled dataset shape {}'.format(Counter(y_res)))


#del train['deal']

#X_full_res = csr_matrix(X_res.drop(['deal_probability'], axis=1))
#X_train_res, X_valid_res, y_train_res, y_valid_res = train_test_split(X_full_res, X_res['deal_probability'], test_size=0.20, random_state=RANDOM_STATE)
###################################

# ## Building a simple model
#
feature_names = list(train_with_features.drop(['deal_probability'], axis=1).columns)
del train_with_features, test_with_features

gc.collect()




def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



print( "Create lgb.datasets...")
start_time = time.time()
dataset_train = lgb.Dataset(X_train, label=y_train,
                            feature_name = feature_names)
dataset_valid = lgb.Dataset(X_valid, label=y_valid,
                            feature_name = feature_names)
print('done in {} seconds.'.format(time.time() - start_time))

#####################################################
#Model0
params = {'learning_rate': 0.04,
          'max_depth': 13,
          'boosting': 'gbdt',#dart, gbdt, goss
          'objective': 'regression',#'regression' #binary, xentropy
          'metric': ['rmse'],#,'auc'
          'is_training_metric': True,
          'seed': RANDOM_STATE,
          'num_leaves': 512,# #default 31
          'feature_fraction': 0.4,#default 1.0
#          'device': 'gpu',
          'bagging_fraction': 0.8,
          'bagging_freq': 5
          }

#print('number of features is ', len(feature_names))
#evals_results = {}
#
#start_time = time.time()
#model0 = lgb.train(params,
#                  dataset_train,
#                  4000,
#                  [
##                  dataset_train,
#                   dataset_valid
#                   ],
#                  [#'train',
#                          'valid'],
#                  evals_result=evals_results,
#                  verbose_eval=50,
#                  early_stopping_rounds=200)
#print('Model training without holdout done in {} seconds.'.format(time.time() - start_time))
#
#model0_store = "leonid_model{:d}_wo_holdout_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_{:.6g}.pkl".format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model0.best_iteration-1]
#           )
#print( "Saving model0 data...")
#print(model0_store)
#start_time = time.time()
#with open(os.path.abspath(model0_store), 'wb') as f:
#    pickle.dump((model0, evals_results), f)
#print('Done in {} seconds.'.format(time.time() - start_time))
#
## feature importances
#model0_fi = pd.concat([pd.Series(model0.feature_name(), name = 'feature'),
#        pd.Series(model0.feature_importance(importance_type='gain'), name = 'gain'),
#        pd.Series(model0.feature_importance(), name = 'importance')
#        ], axis=1).sort_values(by = ['gain'], ascending = False)
#
#print('Save feature importances...')
#start_time = time.time()
#model0_fi.to_csv("model{:d}_fulltrain_{:d}_features_{:.6g}_{}_{}_md{:d}_nl{:d}_ff{:.1g}_fi_gain.csv".format(FILENO, len(feature_names),
#           evals_results['valid']['rmse'][model0.best_iteration-1],
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction']),
#            index=False)
#print('Done in {} seconds.'.format(time.time() - start_time))

model0_store = "leonid_model21_wo_holdout_fulltrain_606_features_regression_gbdt_md13_nl512_ff0.4.pkl"
if os.path.isfile(model0_store):
    print("loading model w/o holdout data from pickle file", model0_store)
    with open(os.path.abspath(model0_store), 'rb') as f:
        model0, evals_results = pickle.load(f, encoding='bytes')
feature_names = list(model0.feature_name())

print("model0.best_iteration: ", model0.best_iteration, "\n",
      'rmse (train, validation):',#train,
#      "{:.6g}".format(evals_results['train']['rmse'][model0.best_iteration-1]),
      "{:.6g}".format(rmse(model0.predict(X_train), y_train)),
      "{:.6g}".format(evals_results['valid']['rmse'][model0.best_iteration-1]))


#print('Plot feature importances...')
#start_time = time.time()
##ax = lgb.plot_importance(model0, max_num_features=250, figsize = (8, 40),
##                         )
##plt.savefig('model{:d}_{}_{}_md{:d}_nl{:d}_fi_{:.6g}.png'.format(FILENO,
##            params['objective'], params['boosting'], params['max_depth'],
##            params['num_leaves'], evals_results['valid']['rmse'][model0.best_iteration-1]), dpi=600,bbox_inches="tight")
##plt.show()
#ax = lgb.plot_importance(model0, max_num_features=250, figsize = (8, 40),
#                         importance_type='gain'
#                         )
#plt.savefig('model{:d}_{}_{}_md{:d}_nl{:d}_fi_gain_{:.6g}.png'.format(FILENO,
#            params['objective'], params['boosting'], params['max_depth'],
#            params['num_leaves'],
#            evals_results['valid']['rmse'][model0.best_iteration-1]
#            ),
#dpi=600,bbox_inches="tight")
#plt.show()
#print('Done in {} seconds.'.format(time.time() - start_time))

#print('Plot metrics recorded during training...')
#ax = lgb.plot_metric(evals_results, metric='rmse', figsize = (10, 8))
#plt.savefig('leonid_model{:d}_wo_holdout_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_metrics_{:.6g}.png'.format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model0.best_iteration-1]), dpi=600,bbox_inches="tight")
#plt.show()


#print(list(fi[fi.gain < 1].feature))
#print('Predict valid with model w/o holdout...')
#pred_valid = model0.predict(X_valid)
#print('metric for validation predict: rmse {:.6g}'.format(rmse(pred_valid, y_valid)))

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
            index=False, float_format="%.5g")

##############
#print("Start learning model on full train data...")
#start_time = time.time()
#model0_f = lgb.train(params,
#                  lgb.Dataset(X_full, label=y_full,
#                              feature_name = feature_names),
#                  model0.best_iteration,
#                  verbose_eval=50)
#print('Model training done in {} seconds.'.format(time.time() - start_time))
#
#model0_f_store = "leonid_model{:d}_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_{:.6g}.pkl".format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model0.best_iteration-1])
#print( "Saving full model data...")
#start_time = time.time()
#with open(os.path.abspath(model0_f_store), 'wb') as f:
#    pickle.dump(model0_f, f)
#print('Done in {} seconds.'.format(time.time() - start_time))


model0_f_store = "leonid_model21_fulltrain_606_features_regression_gbdt_md13_nl512_ff0.4.pkl"
if os.path.isfile(model0_f_store):
    print("loading fully trained model data from pickle file", model0_f_store)
    with open(os.path.abspath(model0_f_store), 'rb') as f:
        model0_f = pickle.load(f, encoding='bytes')



print('Predict test with full model...')
pred = model0_f.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv("../submit/leonid_model{:d}_predict_test_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model0.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")


######################################################
#Model1
params = {'learning_rate': 0.04,
          'max_depth': 13,
          'boosting': 'goss',#dart, gbdt, goss
          'objective': 'xentropy',#'regression' #binary, xentropy
          'metric': ['rmse'],#,'auc'
          'is_training_metric': True,
          'seed': RANDOM_STATE,
          'num_leaves': 512,# #default 31
          'feature_fraction': 0.4,#default 1.0
#          'device': 'gpu',
#          'bagging_fraction': 0.8,
#          'bagging_freq': 5
          }


#print('number of features is ', len(feature_names))
#evals_results = {}
#
#start_time = time.time()
#model1 = lgb.train(params,
#                  dataset_train,
#                  4000,
#                  [
##                  dataset_train,
#                   dataset_valid
#                   ],
#                  [#'train',
#                          'valid'],
#                  evals_result=evals_results,
#                  verbose_eval=50,
#                  early_stopping_rounds=200)
#print('Model training without holdout done in {} seconds.'.format(time.time() - start_time))
#
#model1_store = "leonid_model{:d}_wo_holdout_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_{:.6g}.pkl".format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model1.best_iteration-1]
#           )
#print( "Saving model1 data...")
#print(model1_store)
#start_time = time.time()
#with open(os.path.abspath(model1_store), 'wb') as f:
#    pickle.dump((model1, evals_results), f)
#print('Done in {} seconds.'.format(time.time() - start_time))
#
## feature importances
#model1_fi = pd.concat([pd.Series(model1.feature_name(), name = 'feature'),
#        pd.Series(model1.feature_importance(importance_type='gain'), name = 'gain'),
#        pd.Series(model1.feature_importance(), name = 'importance')
#        ], axis=1).sort_values(by = ['gain'], ascending = False)
#
#print('Save feature importances...')
#start_time = time.time()
#model1_fi.to_csv("model{:d}_fulltrain_{:d}_features_{:.6g}_{}_{}_md{:d}_nl{:d}_ff{:.1g}_fi_gain.csv".format(FILENO, len(feature_names),
#           evals_results['valid']['rmse'][model1.best_iteration-1],
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction']),
#            index=False)
#print('Done in {} seconds.'.format(time.time() - start_time))

model1_store = "leonid_model21_wo_holdout_fulltrain_606_features_xentropy_goss_md13_nl512_ff0.4.pkl"
if os.path.isfile(model1_store):
    print("loading model w/o holdout data from pickle file", model1_store)
    with open(os.path.abspath(model1_store), 'rb') as f:
        model1, evals_results = pickle.load(f, encoding='bytes')
feature_names = list(model1.feature_name())

print("model1.best_iteration: ", model1.best_iteration, "\n",
      'rmse (train, validation):',#train,
#      "{:.6g}".format(evals_results['train']['rmse'][model1.best_iteration-1]),
      "{:.6g}".format(rmse(model1.predict(X_train), y_train)),
      "{:.6g}".format(evals_results['valid']['rmse'][model1.best_iteration-1]))

#print('Plot feature importances...')
#start_time = time.time()
##ax = lgb.plot_importance(model1, max_num_features=250, figsize = (8, 40),
##                         )
##plt.savefig('model{:d}_{}_{}_md{:d}_nl{:d}_fi_{:.6g}.png'.format(FILENO,
##            params['objective'], params['boosting'], params['max_depth'],
##            params['num_leaves'], evals_results['valid']['rmse'][model1.best_iteration-1]), dpi=600,bbox_inches="tight")
##plt.show()
#ax = lgb.plot_importance(model1, max_num_features=250, figsize = (8, 40),
#                         importance_type='gain'
#                         )
#plt.savefig('model{:d}_{}_{}_md{:d}_nl{:d}_fi_gain_{:.6g}.png'.format(FILENO,
#            params['objective'], params['boosting'], params['max_depth'],
#            params['num_leaves'],
#            evals_results['valid']['rmse'][model1.best_iteration-1]
#            ),
#dpi=600,bbox_inches="tight")
#plt.show()
#print('Done in {} seconds.'.format(time.time() - start_time))

#print('Plot metrics recorded during training...')
#ax = lgb.plot_metric(evals_results, metric='rmse', figsize = (10, 8))
#plt.savefig('leonid_model{:d}_wo_holdout_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_metrics_{:.6g}.png'.format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model1.best_iteration-1]), dpi=600,bbox_inches="tight")
#plt.show()


#print(list(fi[fi.gain < 1].feature))
#print('Predict valid with model w/o holdout...')
#pred_valid = model1.predict(X_valid)
#print('metric for validation predict: rmse {:.6g}'.format(rmse(pred_valid, y_valid)))

print('Predict holdout_v2 with model w/o holdout...')
pred_holdout = model1.predict(X_valid)
print('metric for holdout_v2 predict: rmse {:.6g}'.format(rmse(pred_holdout,
      y_valid)))
sub_holdout = pd.concat([pd.Series(y_valid.index.values, name = 'item_id'),
             pd.Series(np.clip(pred_holdout, 0, 1), name = 'deal_probability')],
             axis=1, verify_integrity=True)
#clipping is necessary.

sub_holdout.to_csv("../submit/leonid_model{:d}_predict_holdout_v2_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model1.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")

print('Predict train with model w/o holdout...')
pred_train = model1.predict(X_full)
print('metric for train_orig predict: rmse {:.6g}'.format(rmse(pred_train,
      y_full)))
sub_train = pd.concat([pd.Series(y_full.index.values, name = 'item_id'),
             pd.Series(np.clip(pred_train, 0, 1), name = 'deal_probability')],
        axis=1, verify_integrity=True)
#clipping is necessary.
sub_train.to_csv("../submit/leonid_model{:d}_predict_train_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model1.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")


print('Predict test with model w/o holdout...')
pred = model1.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv("../submit/leonid_model{:d}_holdout_predict_test_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model1.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")

##############
#print("Start learning model on full train data...")
#start_time = time.time()
#model1_f = lgb.train(params,
#                  lgb.Dataset(X_full, label=y_full,
#                              feature_name = feature_names),
#                  model1.best_iteration,
#                  verbose_eval=50)
#print('Model training done in {} seconds.'.format(time.time() - start_time))
#
#model1_f_store = "leonid_model{:d}_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_{:.6g}.pkl".format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model1.best_iteration-1])
#print( "Saving full model data...")
#start_time = time.time()
#with open(os.path.abspath(model1_f_store), 'wb') as f:
#    pickle.dump(model1_f, f)
#print('Done in {} seconds.'.format(time.time() - start_time))


model1_f_store = "leonid_model21_fulltrain_606_features_xentropy_goss_md13_nl512_ff0.4.pkl"
if os.path.isfile(model1_f_store):
    print("loading fully trained model data from pickle file", model1_f_store)
    with open(os.path.abspath(model1_f_store), 'rb') as f:
        model1_f = pickle.load(f, encoding='bytes')



print('Predict test with full model...')
pred = model1_f.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv("../submit/leonid_model{:d}_predict_test_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model1.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")




######################################################
#Model2
params = {'learning_rate': 0.04,
          'max_depth': 13,
          'boosting': 'gbdt',#dart, gbdt, goss
          'objective': 'xentropy',#'regression' #binary, xentropy
          'metric': ['rmse'],#,'auc'
          'is_training_metric': True,
          'seed': RANDOM_STATE,
          'num_leaves': 512,# #default 31
          'feature_fraction': 0.4,#default 1.0
#          'device': 'gpu',
          'bagging_fraction': 0.8,
          'bagging_freq': 5
          }


#print('number of features is ', len(feature_names))
#evals_results = {}
#
#start_time = time.time()
#model2 = lgb.train(params,
#                  dataset_train,
#                  4000,
#                  [
##                  dataset_train,
#                   dataset_valid
#                   ],
#                  [#'train',
#                          'valid'],
#                  evals_result=evals_results,
#                  verbose_eval=50,
#                  early_stopping_rounds=200)
#print('Model training without holdout done in {} seconds.'.format(time.time() - start_time))
#
#model2_store = "leonid_model{:d}_wo_holdout_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_{:.6g}.pkl".format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model2.best_iteration-1]
#           )
#print( "Saving model2 data...")
#print(model2_store)
#start_time = time.time()
#with open(os.path.abspath(model2_store), 'wb') as f:
#    pickle.dump((model2, evals_results), f)
#print('Done in {} seconds.'.format(time.time() - start_time))
#
## feature importances
#model2_fi = pd.concat([pd.Series(model2.feature_name(), name = 'feature'),
#        pd.Series(model2.feature_importance(importance_type='gain'), name = 'gain'),
#        pd.Series(model2.feature_importance(), name = 'importance')
#        ], axis=1).sort_values(by = ['gain'], ascending = False)
#
#print('Save feature importances...')
#start_time = time.time()
#model2_fi.to_csv("model{:d}_fulltrain_{:d}_features_{:.6g}_{}_{}_md{:d}_nl{:d}_ff{:.1g}_fi_gain.csv".format(FILENO, len(feature_names),
#           evals_results['valid']['rmse'][model2.best_iteration-1],
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction']),
#            index=False)
#print('Done in {} seconds.'.format(time.time() - start_time))

model2_store = "leonid_model21_wo_holdout_fulltrain_606_features_xentropy_gbdt_md13_nl512_ff0.4.pkl"
if os.path.isfile(model2_store):
    print("loading model w/o holdout data from pickle file", model2_store)
    with open(os.path.abspath(model2_store), 'rb') as f:
        model2, evals_results = pickle.load(f, encoding='bytes')
feature_names = list(model2.feature_name())

print("model2.best_iteration: ", model2.best_iteration, "\n",
      'rmse (train, validation):',#train,
#      "{:.6g}".format(evals_results['train']['rmse'][model2.best_iteration-1]),
      "{:.6g}".format(rmse(model2.predict(X_train), y_train)),
      "{:.6g}".format(evals_results['valid']['rmse'][model2.best_iteration-1]))

#print('Plot feature importances...')
#start_time = time.time()
##ax = lgb.plot_importance(model2, max_num_features=250, figsize = (8, 40),
##                         )
##plt.savefig('model{:d}_{}_{}_md{:d}_nl{:d}_fi_{:.6g}.png'.format(FILENO,
##            params['objective'], params['boosting'], params['max_depth'],
##            params['num_leaves'], evals_results['valid']['rmse'][model2.best_iteration-1]), dpi=600,bbox_inches="tight")
##plt.show()
#ax = lgb.plot_importance(model2, max_num_features=250, figsize = (8, 40),
#                         importance_type='gain'
#                         )
#plt.savefig('model{:d}_{}_{}_md{:d}_nl{:d}_fi_gain_{:.6g}.png'.format(FILENO,
#            params['objective'], params['boosting'], params['max_depth'],
#            params['num_leaves'],
#            evals_results['valid']['rmse'][model2.best_iteration-1]
#            ),
#dpi=600,bbox_inches="tight")
#plt.show()
#print('Done in {} seconds.'.format(time.time() - start_time))

#print('Plot metrics recorded during training...')
#ax = lgb.plot_metric(evals_results, metric='rmse', figsize = (10, 8))
#plt.savefig('leonid_model{:d}_wo_holdout_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_metrics_{:.6g}.png'.format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model2.best_iteration-1]), dpi=600,bbox_inches="tight")
#plt.show()


#print(list(fi[fi.gain < 1].feature))
#print('Predict valid with model w/o holdout...')
#pred_valid = model2.predict(X_valid)
#print('metric for validation predict: rmse {:.6g}'.format(rmse(pred_valid, y_valid)))

print('Predict holdout_v2 with model w/o holdout...')
pred_holdout = model2.predict(X_valid)
holdout_metric_result = rmse(pred_holdout, y_valid)
print('metric for holdout_v2 predict: rmse {:.6g}'.format(holdout_metric_result))
sub_holdout = pd.concat([pd.Series(y_valid.index.values, name = 'item_id'),
             pd.Series(np.clip(pred_holdout, 0, 1), name = 'deal_probability')],
             axis=1, verify_integrity=True)
#clipping is necessary.

#sub_holdout.to_csv("leonid_model{:d}_{}_{}_predict_holdout_{:.6g}_float_format_5g.csv".format(FILENO,
#           params['objective'], params['boosting'], holdout_metric_result),
#            index=False, float_format="%.5g")
#sub_holdout.to_csv("leonid_model{:d}_{}_{}_predict_holdout_{:.6g}_float_format_6g.csv".format(FILENO,
#           params['objective'], params['boosting'], holdout_metric_result),
#            index=False, float_format="%.6g")
#sub_holdout.to_csv("leonid_model{:d}_{}_{}_predict_holdout_{:.6g}_float_format_7g.csv".format(FILENO,
#           params['objective'], params['boosting'], holdout_metric_result),
#            index=False, float_format="%.7g")
#sub_holdout.to_csv("leonid_model{:d}_{}_{}_predict_holdout_{:.6g}_float_format_8g.csv".format(FILENO,
#           params['objective'], params['boosting'], holdout_metric_result),
#            index=False, float_format="%.8g")
#sub_holdout.to_csv("leonid_model{:d}_{}_{}_predict_holdout_{:.6g}_float_format_9g.csv".format(FILENO,
#           params['objective'], params['boosting'], holdout_metric_result),
#            index=False, float_format="%.9g")
#sub_holdout.to_csv("leonid_model{:d}_{}_{}_predict_holdout_{:.6g}_float_format_not_set.csv".format(FILENO,
#           params['objective'], params['boosting'], holdout_metric_result),
#            index=False)






sub_holdout.to_csv("../submit/leonid_model{:d}_predict_holdout_v2_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model2.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")

print('Predict train with model w/o holdout...')
pred_train = model2.predict(X_full)
print('metric for train_orig predict: rmse {:.6g}'.format(rmse(pred_train,
      y_full)))
sub_train = pd.concat([pd.Series(y_full.index.values, name = 'item_id'),
             pd.Series(np.clip(pred_train, 0, 1), name = 'deal_probability')],
        axis=1, verify_integrity=True)
#clipping is necessary.
sub_train.to_csv("../submit/leonid_model{:d}_predict_train_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model2.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")


print('Predict test with model w/o holdout...')
pred = model2.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv("../submit/leonid_model{:d}_holdout_predict_test_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model2.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")

##############
#print("Start learning model on full train data...")
#start_time = time.time()
#model2_f = lgb.train(params,
#                  lgb.Dataset(X_full, label=y_full,
#                              feature_name = feature_names),
#                  model2.best_iteration,
#                  verbose_eval=50)
#print('Model training done in {} seconds.'.format(time.time() - start_time))
#
#model2_f_store = "leonid_model{:d}_fulltrain_{:d}_features_{}_{}_md{:d}_nl{:d}_ff{:.1g}_{:.6g}.pkl".format(FILENO, len(feature_names),
#           params['objective'], params['boosting'], params['max_depth'],
#           params['num_leaves'], params['feature_fraction'],
#           evals_results['valid']['rmse'][model2.best_iteration-1])
#print( "Saving full model data...")
#start_time = time.time()
#with open(os.path.abspath(model2_f_store), 'wb') as f:
#    pickle.dump(model2_f, f)
#print('Done in {} seconds.'.format(time.time() - start_time))


model2_f_store = "leonid_model21_fulltrain_606_features_xentropy_gbdt_md13_nl512_ff0.4.pkl"
if os.path.isfile(model2_f_store):
    print("loading fully trained model data from pickle file", model2_f_store)
    with open(os.path.abspath(model2_f_store), 'rb') as f:
        model2_f = pickle.load(f, encoding='bytes')



print('Predict test with full model...')
pred = model2_f.predict(X_test_full)
#clipping is necessary.
sub['deal_probability'] = np.clip(pred, 0, 1)
sub.to_csv("../submit/leonid_model{:d}_predict_test_fulltrain_{:d}_features_{:.6g}_{}_{}.csv".format(FILENO, len(feature_names),
           evals_results['valid']['rmse'][model2.best_iteration-1],
           params['objective'], params['boosting']),
            index=False, float_format="%.5g")
