""""Formats the data used for training in pre-processing step"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # stratified split for data

def get_text_to_gender():
    df = get_sonnets_with_authors_filtered()
    # return X train, X test, y train, y test
    return train_test_split(df['content'], df['gender'], test_size=0.1, stratify=df['gender'], random_state=1)

def get_text_to_period():
    df = get_sonnets_with_authors_filtered()
    df = df[['content', 'normdate']]
    df['normdate'] = df['normdate'].apply(np.floor)
    return train_test_split(df['content'], df['normdate'], test_size=0.1, stratify=df['normdate'], random_state=1)

def get_text_to_country_of_origin():
    df = get_sonnets_with_authors_filtered()
    return train_test_split(df['content'], df['country-birth'], test_size=0.1, stratify=df['country-birth'], random_state=1)

def get_sonnets():
    """"Returns dataframe of sonnets with cols aid (to index author) and content (text of poem)
    """
    # path is /data/disco_files/txt/<period>/per-sonnet
    sonnets = []
    data_path = Path('../data/disco_files/txt')
    for period_path in data_path.iterdir():
        if period_path.is_dir():
            for selection in period_path.iterdir():
                if selection.is_dir() and selection.name == 'per-sonnet':
                    for sonnet_file in selection.iterdir():
                        with open(sonnet_file, 'r', encoding='utf-8') as f:
                            sonnets.append({'aid': sonnet_file.name.split('_')[0][5:], 'content': f.read()})

    return pd.DataFrame(sonnets)

def get_authors(cols):
    author_path = '../data/disco_files/author_metadata.tsv'
    author_df = pd.read_csv(author_path, sep='\t')
    return author_df[cols]

def get_sonnets_with_authors_filtered():
    cols = ['aid', 'normdate', 'gender', 'first-name', 'preposition-name', 'second-name', 'country-birth']
    author_df = get_authors(cols)
    
    all_df = get_sonnets().merge(author_df, how='left', on='aid')
    all_df = all_df[all_df['aid'].notna() & all_df['normdate'].notna() & all_df['gender'].notna() & all_df['country-birth'].notna()]

    return all_df