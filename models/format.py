""""Formats the data used for training in pre-processing step"""
from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def get_text_to_gender():
    df = get_sonnets_with_authors_filtered()
    # return X train, X test, y train, y test
    return train_test_split(df[['content','gender', 'rhyme', 'met']], 'gender', test_size=0.1, random_state=1)

def get_text_to_period():
    df = get_sonnets_with_authors_filtered()
    df = df[['content', 'normdate', 'rhyme', 'met']]
    df['normdate'] = df['normdate'].apply(np.floor)
    return train_test_split(df[['content','normdate', 'rhyme', 'met']], 'normdate', test_size=0.1, random_state=1)

def get_periods():
    df = get_sonnets_with_authors_filtered()
    return list(df['normdate'].apply(np.floor).unique())

def get_text_to_country_of_origin():
    df = get_sonnets_with_authors_filtered()
    return train_test_split(df[['content','country-birth', 'rhyme', 'met']], 'country-birth', test_size=0.1, random_state=1)

def get_countries():
    df = get_sonnets_with_authors_filtered()
    return list(df['country-birth'].unique())

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
                        # extract rhymes, etc from tei
                        line_rhyme, line_meter = get_rhyme_meter(period_path.name + '/' + selection.name + '/' + sonnet_file.name)
                        if line_meter is None or line_rhyme is None:
                            continue
                        with open(sonnet_file, 'r', encoding='utf-8') as f:
                            sonnets.append({'aid': sonnet_file.name.split('_')[0][5:], 'content': f.read().replace('\n\n', '\n'), 'met':line_meter, 'rhyme':line_rhyme})

    return pd.DataFrame(sonnets)

namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
def get_rhyme_meter(sonnet_file_name):
    try:
        tree = ET.parse(f'../data/disco_files/tei/{sonnet_file_name[:-3]}xml')
    except Exception:
        return None, None
    root = tree.getroot()
    lines = root.findall('.//tei:l', namespace)
    r = [line.get('rhyme') for line in lines][:14] # cut at 14
    m = [line.get('met') for line in lines]
    if any(x is None for x in r) or any(x is None for x in m):
        return None, None
    return r, m

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

# sklearn inspired stratified train_test_split function
def train_test_split(df, stratify_col, test_size=0.1, random_state=None):
    if stratify_col not in df.columns:
        raise ValueError(f"'{stratify_col}' is not a column in the DataFrame.")
    
    np.random.seed(random_state)
    df_train_list = []
    df_test_list = []

    # Group by the stratify column
    for label, group in df.groupby(stratify_col):
        n = len(group)
        if isinstance(test_size, float):
            n_test = int(np.floor(test_size * n))
        else:
            n_test = min(test_size, n)
        
        shuffled_indices = np.random.permutation(group.index)
        test_indices = shuffled_indices[:n_test]
        train_indices = shuffled_indices[n_test:]
        
        df_test_list.append(df.loc[test_indices])
        df_train_list.append(df.loc[train_indices])
    
    df_train = pd.concat(df_train_list).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_test = pd.concat(df_test_list).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_train, df_test
