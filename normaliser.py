#!/usr/bin/env python3.7
import pandas as pd
import numpy as np

def get_mu_sigma(df_feat, columns):
    
    # computes mu and sigma for train dataset
    # return:
    # mu - dict with feature:mu key:values
    # sigma - dict with feature:sigma key:values
    
    mu = {}
    sigma = {}
    
    for column in columns[1:]:
        mu[column] = df_feat[column].values.mean().round(2)
        sigma[column] = df_feat[column].values.std().round(2)
    
    return mu, sigma

def get_z_score(df_feat, feat_set_name, columns, mu, sigma):
    
    # computes z-scores and stores to newly created dataframe
    # return:
    # z_score - dataframe with z-scores for initial test file values
    
    z_score = pd.DataFrame()
    
    for column in columns[2:]:
        z_score[column] = df_feat[column].apply(lambda x: (x-mu[column])/sigma[column]).astype('float32')
    
    z_score.columns = ['_'.join(['feature', feat_set_name, 'stand', str(i)]) for i in range(1, len(columns)-1)]
    
    return z_score

def normalise(df_feat, feat_set_name, columns, mu, sigma, norm_type):
    
    # calls normalisation function, specified by norm_type keyword
    # return:
    # normalised - dataframe with applied normalisation

    norm_types = {'z_score': get_z_score}
    normalised =  norm_types[norm_type](df_feat, feat_set_name, columns, mu, sigma)
    
    return normalised

def expand_features(data, columns):
    
    # expands the data into full dataframe, applies column/feature names
    # return:
    # df_feat - initially processed dataframe
    
    df_feat = data['features'].str.split(',', expand=True).astype('int16')
    df_feat = pd.concat([data.id_job, df_feat], axis=1)
    df_feat.columns = columns
    
    return df_feat

def get_init_info(train_file):
    
    # reads the data, gets data, column/feature names and
    # actual feature set name
    # return:
    # data - initial train dataset
    # columns - expanded dataset column names
    # feat_set_name - actual feature set name
    
    data = pd.read_csv(train_file, sep='\t')
    
    features = data.features[0].split(',')
    feat_set_name = features[0]
    columns = ['id_job', 'feature_set_code']
    for i in range(1, len(features[1:])+1):
        columns.append('feature_' + feat_set_name + '_' + str(i))
        
    return data, columns, feat_set_name

def transform_test_data(test_file, train_file, norm_type):
    
    # gets the initial train set data with chosen statistics,
    # reads a test dataset in chunks, applies preferred normalisation,
    # adds two aggregated features and iteratively writes
    # the result to 'test_proc.tsv' file
        
    data, columns, feat_set_name = get_init_info(train_file)
    df_feat = expand_features(data, columns)
    mu, sigma = get_mu_sigma(df_feat, columns)
    
    reader = pd.read_table(test_file, sep='\t', chunksize=100000)
    for chunk in reader:
    
        df_feat = expand_features(chunk, columns)
        add_feat = pd.DataFrame(columns=['max_feature_2_index', 'max_feature_2_abs_mean_diff'])
        for i, row in df_feat.iterrows():
            max_value = row[1:].values.max()
            max_index = row[1:][row[1:] == max_value].index.values[0]
            add_feat.loc[i, 'max_feature_2_index'] = int(max_index.split('_')[2])
            add_feat.loc[i, 'max_feature_2_abs_mean_diff'] = abs(max_value - mu[max_index])

        df_norm = normalise(df_feat, feat_set_name, columns, mu, sigma, norm_type)

        df_test = pd.concat([df_feat.iloc[:, 0:2], 
                         df_norm, 
                         add_feat], axis=1)

        df_test.to_csv('test_proc.tsv', sep='\t', index=False, mode='a')

