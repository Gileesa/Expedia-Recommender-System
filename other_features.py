
#
#
#

import pandas as pd
import numpy as np

def add_search_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds within-search relative features, comparing each hotel
    against other hotels shown in the same search.
    """
    # get some pricing stats per search
    search_price_mean = df.groupby('srch_id')['price_usd'].transform('mean') # mean price in search
    search_price_std = df.groupby('srch_id')['price_usd'].transform('std') # std of prices in search

    # relative price features
    df['price_pct_rank'] = df.groupby('srch_id')['price_usd'].rank(pct=True) # rank prices
    df['price_usd_diff'] = df['price_usd'] - search_price_mean # get difference from mean
    df['price_usd_zscore'] = df['price_usd_diff'] / search_price_std # how many standard deviations a value is away from the mean
    df['price_per_night'] = df['price_usd'] / df['srch_length_of_stay'].clip(lower=1) # price per night
    df['price_per_person'] = df['price_usd'] / df['srch_adults_count'].clip(lower=1) # price per person

    # get some star rating stats per search
    search_star_mean = df.groupby('srch_id')['prop_starrating'].transform('mean') #
    search_star_std = df.groupby('srch_id')['prop_starrating'].transform('std')

    # star rating features
    df['prop_starrating_diff'] = df['prop_starrating'] - search_star_mean
    df['prop_starrating_zscore'] = df['prop_starrating_diff'] / search_star_std

    # get some search review stats per search
    search_review_mean = df.groupby('srch_id')['prop_review_score'].transform('mean')
    search_review_std = df.groupby('srch_id')['prop_review_score'].transform('std')

    # review score features
    df['prop_review_score_diff'] = df['prop_review_score'] - search_review_mean
    df['prop_review_score_zscore'] = df['prop_review_score_diff'] / search_review_std

    # position features (only available in train/valid, not test)
    # we take the log
    if 'position' in df.columns:
        df['log_position'] = np.log1p(df['position'])

    return df