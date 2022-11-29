import numpy as np
import pandas as pd
import re
import os
from sklearn.model_selection import StratifiedKFold


def clean_record(doc):
    doc = doc.lower()
    doc = re.sub('[^a-z0-9.,\'/]', ' ', doc)
    doc = re.sub('xx+|\*+', ' ', doc)
    doc = re.sub(' +', ' ', doc)
    return doc


def load_df_data(df, narr_col, out_col, ids_col=None, min_stc_len=20):
    df = df[~df[out_col].isna()]  # Clean NA in output column
    narr_len = np.array([len(x) for x in df[narr_col]])
    df = df[narr_len > min_stc_len]
    df.reset_index(drop=True, inplace=True)
    docs = df[narr_col].astype(str)
    docs = [clean_record(doc) for doc in docs]
    labels = df[out_col].values.astype(int)
    unique_ids = df[ids_col].values if ids_col is not None else np.arange(len(docs))
    return docs, labels, unique_ids


def load_csv_data(path, narr_col, out_col, ids_col=None, min_stc_len=20):
    df = pd.read_csv(path)
    return load_df_data(df, narr_col, out_col, ids_col, min_stc_len)


def split_data(docs, labels):
    train_idx, test_idx = stratified_kfold_split(labels, n_splits=4, seed=0)[1]
    train_docs, test_docs = list(np.array(docs)[train_idx]), list(np.array(docs)[test_idx])
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    return train_docs, test_docs, train_labels, test_labels


def comput_max_stc_len(docs, percentile=95):
    narr_len = np.array([len(x.split()) for x in docs])
    return int(np.percentile(np.array(narr_len), percentile).astype(int))
 

def stratified_kfold_split(y, n_splits=4, shuffle=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    idx_folds = []
    for x in skfolds.split(np.zeros(len(y)), y):
        train_idx = np.random.permutation(x[0]) if shuffle else x[0]
        test_idx = np.random.permutation(x[1]) if shuffle else x[1]
        idx_folds.append([train_idx, test_idx])
    return idx_folds


def read_file(file_path):
    with open(file_path) as f:
        content = f.read()
        return content


def write_file(file_path, contents=None):
    with open(file_path, 'w') as f:
        f.write(contents)
        

def expand_dict(dict_, key1, key2=None):
    if key2 is None:
        return [j for i in dict_ for j in i[key1]]
    else:
        return [j for i in dict_ for j in i[key1][key2]]

    
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)