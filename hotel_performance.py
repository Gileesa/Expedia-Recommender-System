#
# Code where we generate features for the hotel's performance
# mostly based on data that is only available for the training set
#

# git pull origin gileesa2 --no-rebase


# We are trying to predict booking_bool -> make sure no leakage!

import pandas as pd
from pandas import Series


# # Open training set
# train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)

# # open test set
# test_df = pd.read_csv('test_set_VU_DM.csv', low_memory=False)

def extract_hotel_performance_train(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:    
    """
    Function that adds features regarding hotel profiles to training data and a separate
    hotel_performance dataframe that can be merged into the test data.

    TRAINING DF
    -----------
    Function that extracts hotel performance features from training data based on prop_id.
    To prevent data leakage, the 'current' row is left out (leave-one-out) for the training data).

    HOTEL_PERFORMANCE DF
    --------------------
    This function also builts a dataframe containing hotel profiles (hotel_performance), which can
    be merged into the test data (not in this function). This does not involve LOO

    BAYESIAN SMOOTHING
    ------------------
    Bayesian smoothing is a technique used to handle sparse data. Essentially,
    some statistics need to be smoothed because their case does not appear often
    enough in the data to be reliable, e.g. a hotel booked 2 out of 2 times looks
    perfect but is unreliable with only 2 observations.

    This function makes use of Bayesian smoothing in two ways:

    - Smoothing destination statistics towards the global stats (handles sparse destinations)
    - Smoothing hotel statistics towards the smoothed destination stats (handles sparse hotels)

    This creates a three-level hierarchy: global mean -> destination mean -> hotel rate,
    where each level is only trusted proportionally to how much data supports it.

    Args:
        - train_df (pd.DataFrame): the training pandas dataframe.

    Returns:
        - train_df (pd.DataFrame): the train dataframe with new features.
        - hotel_performance (pd.DataFrame): the dataframe containing hotel profiles built from training data.
    """

    print(f"\n{'='*50}")
    print(f"INPUT: train_df shape: {train_df.shape}")
    print(f"INPUT: unique prop_ids: {train_df['prop_id'].nunique()}")
    print(f"INPUT: unique srch_ids: {train_df['srch_id'].nunique()}")
    print(f"INPUT: unique destinations: {train_df['srch_destination_id'].nunique()}")

    # Get global stats
    global_position_avg = train_df['position'].mean()
    global_booking_rate = train_df['booking_bool'].mean()
    global_click_rate = train_df['click_bool'].mean()

    print(f"\nGLOBAL STATS:")
    print(f"  global_booking_rate: {global_booking_rate:.4f}")
    print(f"  global_click_rate: {global_click_rate:.4f}")
    print(f"  global_position_avg: {global_position_avg:.4f}")

    # Get global destination stats
    dest_stats = train_df.groupby('srch_destination_id').agg(
        dest_bookings=('booking_bool', 'sum'),  
        dest_clicks=('click_bool', 'sum'),      
        dest_count=('booking_bool', 'count')
    ).reset_index()

    print(f"\nDEST_STATS:")
    print(f"  shape: {dest_stats.shape}")
    print(f"  duplicate destinations: {dest_stats['srch_destination_id'].duplicated().sum()}")

    # Get destination stats for one search_id
    dest_search = train_df.groupby(['srch_destination_id', 'srch_id']).agg(
        search_dest_bookings=('booking_bool', 'sum'),
        search_dest_clicks=('click_bool', 'sum'),
        search_dest_count=('booking_bool', 'count')
    ).reset_index()

    print(f"\nDEST_SEARCH:")
    print(f"  shape: {dest_search.shape}")
    print(f"  duplicate (dest, srch) pairs: {dest_search.duplicated(subset=['srch_destination_id', 'srch_id']).sum()}")

    # merge into training set
    rows_before = len(train_df)
    train_df = train_df.merge(
        dest_search,
        on=['srch_destination_id', 'srch_id'],
        how='left'
    )
    print(f"\nAFTER MERGE dest_search: {rows_before} -> {len(train_df)} rows (diff: {len(train_df) - rows_before})")
    print(f"  NaNs in search_dest_bookings: {train_df['search_dest_bookings'].isna().sum()}")

    # also merge total in training set
    rows_before = len(train_df)
    train_df = train_df.merge(
        dest_stats,
        on='srch_destination_id',
        how='left'
    )
    print(f"AFTER MERGE dest_stats: {rows_before} -> {len(train_df)} rows (diff: {len(train_df) - rows_before})")
    print(f"  NaNs in dest_bookings: {train_df['dest_bookings'].isna().sum()}")

    # CALCULATE LOO STATS
    # i.e leaving out current search id
    loo_dest_bookings = (
        train_df['dest_bookings'] # global
        - train_df['search_dest_bookings'] # search_id
    )

    loo_dest_clicks = (
        train_df['dest_clicks']
        - train_df['search_dest_clicks']
    )

    loo_dest_count = (
        train_df['dest_count']
        - train_df['search_dest_count']
    )

    print(f"\nLOO DEST STATS:")
    print(f"  negative loo_dest_count: {(loo_dest_count < 0).sum()}")
    print(f"  zero loo_dest_count: {(loo_dest_count == 0).sum()}")

    # apply Bayesian smoothing to dest stats ( bayesian smoothing: https://en.wikipedia.org/wiki/Bayesian_average )
    # i.e a weighted mean (destination and global stats)
    C_bayesian = 30

    train_df['dest_booking_rate'] = (
        loo_dest_bookings + C_bayesian * global_booking_rate
    ) / (loo_dest_count + C_bayesian)

    train_df['dest_click_rate'] = (
        loo_dest_clicks + C_bayesian * global_click_rate
    ) / (loo_dest_count + C_bayesian)

    print(f"\nDEST RATES:")
    print(f"  dest_booking_rate NaNs: {train_df['dest_booking_rate'].isna().sum()}")
    print(f"  dest_booking_rate range: [{train_df['dest_booking_rate'].min():.4f}, {train_df['dest_booking_rate'].max():.4f}]")

    # merge into train data
    # cols = ['srch_destination_id', 'dest_booking_rate', 'dest_click_rate']
    # train_df = train_df.merge(dest_stats[cols], on='srch_destination_id', how='left')

    # Get hotel-wise stats
    # global
    hotel_total = train_df.groupby('prop_id').agg(
        hotel_bookings=('booking_bool', 'sum'),
        hotel_clicks=('click_bool', 'sum'),
        hotel_position=('position', 'sum'),
        hotel_count=('booking_bool', 'count') # number of appearances
    )

    # per search id
    hotel_search = train_df.groupby(['prop_id', 'srch_id']).agg(
        search_hotel_bookings=('booking_bool', 'sum'),
        search_hotel_clicks=('click_bool', 'sum'),
        search_hotel_position=('position', 'sum'),
        search_hotel_count=('booking_bool', 'count')
    ).reset_index()

    print(f"\nHOTEL_SEARCH:")
    print(f"  shape: {hotel_search.shape}")
    print(f"  duplicate (prop, srch) pairs: {hotel_search.duplicated(subset=['prop_id', 'srch_id']).sum()}")

    # merge into training df
    rows_before = len(train_df)
    train_df = train_df.merge(
        hotel_search,
        on=['prop_id', 'srch_id'],
        how='left'
    )
    print(f"\nAFTER MERGE hotel_search: {rows_before} -> {len(train_df)} rows (diff: {len(train_df) - rows_before})")
    print(f"  NaNs in search_hotel_bookings: {train_df['search_hotel_bookings'].isna().sum()}")

    rows_before = len(train_df)
    train_df = train_df.merge(
        hotel_total,
        on='prop_id',
        how='left'
    )
    print(f"AFTER MERGE hotel_total: {rows_before} -> {len(train_df)} rows (diff: {len(train_df) - rows_before})")
    print(f"  NaNs in hotel_bookings: {train_df['hotel_bookings'].isna().sum()}")

    # apply LOO per search_id
    loo_hotel_bookings = (
        train_df['hotel_bookings']
        - train_df['search_hotel_bookings']
    )

    loo_hotel_clicks = (
        train_df['hotel_clicks']
        - train_df['search_hotel_clicks']
    )

    loo_hotel_position = (
        train_df['hotel_position']
        - train_df['search_hotel_position']
    )

    loo_hotel_count = (
        train_df['hotel_count']
        - train_df['search_hotel_count']
    )

    print(f"\nLOO HOTEL STATS:")
    print(f"  negative loo_hotel_count: {(loo_hotel_count < 0).sum()}")
    print(f"  zero loo_hotel_count: {(loo_hotel_count == 0).sum()}")
    print(f"  NaN loo_hotel_count: {loo_hotel_count.isna().sum()}")

    # train_df should have these for training the ML
    # we apply leave-one-out
    # smoothing towards destination
    train_df['hotel_booking_rate'] = (
        loo_hotel_bookings
        + C_bayesian * train_df['dest_booking_rate']
    ) / (loo_hotel_count + C_bayesian)

    train_df['hotel_click_rate'] = (
        loo_hotel_clicks
        + C_bayesian * train_df['dest_click_rate']
    ) / (loo_hotel_count + C_bayesian)

    train_df['hotel_avg_position'] = (
        loo_hotel_position
        + C_bayesian * global_position_avg
    ) / (loo_hotel_count + C_bayesian)

    train_df['hotel_n_appearances'] = loo_hotel_count

    print(f"\nFINAL HOTEL FEATURES:")
    print(f"  hotel_booking_rate NaNs: {train_df['hotel_booking_rate'].isna().sum()}")
    print(f"  hotel_booking_rate range: [{train_df['hotel_booking_rate'].min():.4f}, {train_df['hotel_booking_rate'].max():.4f}]")
    print(f"  hotel_click_rate NaNs: {train_df['hotel_click_rate'].isna().sum()}")
    print(f"  hotel_n_appearances NaNs: {train_df['hotel_n_appearances'].isna().sum()}")
    print(f"  hotel_n_appearances range: [{train_df['hotel_n_appearances'].min():.0f}, {train_df['hotel_n_appearances'].max():.0f}]")
    print(f"  train_df shape after all merges: {train_df.shape}")


    # drop unnecessary columns
    train_df = train_df.drop(columns=[
        'dest_bookings',
        'dest_clicks',
        'dest_count',

        'search_dest_bookings',
        'search_dest_clicks',
        'search_dest_count',
    ])

    train_df = train_df.drop(columns=[
        'hotel_bookings',
        'hotel_clicks',
        'hotel_position',
        'hotel_count',

        'search_hotel_bookings',
        'search_hotel_clicks',
        'search_hotel_position',
        'search_hotel_count'
    ])

    # ===== FOR THE TEST SET ==========
    # store per prop_id, i.e make profile per hotel
    # we don't apply leave-one-out; use full training data
    hotel_performance = train_df.groupby('prop_id').agg(
        total_bookings=('booking_bool', 'sum'),
        total_clicks=('click_bool', 'sum'),
        total_position=('position', 'sum'),
        total_count=('booking_bool', 'count') # number of appearances
    ).reset_index()

    # Some hotels might have multiple destinations; take most frequent one
    hotel_destination = train_df.groupby('prop_id')['srch_destination_id'].agg(
        lambda x: x.mode()[0]  # most frequent destination for this hotel
    ).reset_index()

    # Merge destination into hotel_performance
    hotel_performance = hotel_performance.merge(hotel_destination, on='prop_id', how='left')

    # Merge smoothed destination rates into hotel_performance
    # non-LOO dest stats:
    dest_performance = pd.DataFrame({
        'srch_destination_id': dest_stats['srch_destination_id'],
        'dest_booking_rate': (
            dest_stats['dest_bookings'] + C_bayesian * global_booking_rate
        ) / (dest_stats['dest_count'] + C_bayesian),

        'dest_click_rate': (
            dest_stats['dest_clicks'] + C_bayesian * global_click_rate
        ) / (dest_stats['dest_count'] + C_bayesian)
    })
    
    # merge:
    hotel_performance = hotel_performance.merge(
        dest_performance,
        on='srch_destination_id',
        how='left'
    )

    # Apply Bayesian smoothing using destination rate
    # no LOO
    hotel_performance['hotel_booking_rate'] = (
        hotel_performance['total_bookings'] + C_bayesian * hotel_performance['dest_booking_rate']
    ) / (hotel_performance['total_count'] + C_bayesian)

    hotel_performance['hotel_click_rate'] = (
        hotel_performance['total_clicks'] + C_bayesian * hotel_performance['dest_click_rate']
    ) / (hotel_performance['total_count'] + C_bayesian)

    hotel_performance['hotel_avg_position'] = (
        hotel_performance['total_position'] + C_bayesian * global_position_avg
    ) / (hotel_performance['total_count'] + C_bayesian)

    hotel_performance['hotel_n_appearances'] = hotel_performance['total_count']

    # Drop unnecessary columns
    hotel_performance = hotel_performance.drop(columns=[
        'total_bookings', 'total_clicks', 'total_position', 'total_count',
        'srch_destination_id', 'dest_booking_rate', 'dest_click_rate'
    ])

    # NOTE: bayesian smoothing: https://jedleee.medium.com/bayesian-laplace-smoothing-applications-in-modern-machine-learning-ef6c38153940 

    return train_df, hotel_performance, dest_performance

def extract_hotel_performance_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Function that extracts hotel performance features from training data
    to merge into test data based on prop_id.
    If a new prop_id appears in the test data, inpute the destination mean or 
    the global mean.
    Most technical features are performed in the extract_hotel_performance_train function

    Params:
    - train_df (pd.DataFrame): the training pandas dataframe (raw)
    - test_df (pd.DataFrame): the test pandas dataframe (raw)
    Returns:
    - train_df (pd.DataFrame): the training pandas dataframe with new features
    - test_df (pd.DataFrame): the test dataframe with new features
    '''

    print(f"\n{'='*50}")
    print(f"TEST INPUT: test_df shape: {test_df.shape}")
    print(f"TEST INPUT: unique srch_ids in test: {test_df['srch_id'].nunique()}")

    # extract features from training_df
    # this includes Bayesian smoothing
    train_df, hotel_performance, dest_performance = extract_hotel_performance_train(train_df)
    print("=======HOTEL PERFORMANCE COLUMNS======")
    print(hotel_performance.columns.tolist())
    print(hotel_performance.shape)
    print("=" * 50)

    print(f"\nHOTEL_PERFORMANCE:")
    print(f"  shape: {hotel_performance.shape}")
    print(f"  duplicate prop_ids: {hotel_performance['prop_id'].duplicated().sum()}")
    print(f"  NaNs: {hotel_performance.isna().sum().sum()}")

    print(f"\nDEST_PERFORMANCE:")
    print(f"  shape: {dest_performance.shape}")
    print(f"  duplicate destinations: {dest_performance['srch_destination_id'].duplicated().sum()}")

    # Get destination mean
    # TODO: smoothing this!
    dest_stats = train_df.groupby('srch_destination_id').agg(
        dest_booking_rate=('booking_bool', 'mean'),
        dest_click_rate=('click_bool', 'mean')
    ).reset_index()

    # Merge into test set
    # uses smoothed destinations !
    rows_before = len(test_df)
    test_df = test_df.merge(hotel_performance, on='prop_id', how='left')
    print(f"\nAFTER MERGE hotel_performance into test: {rows_before} -> {len(test_df)} rows (diff: {len(test_df) - rows_before})")
    print(f"  new hotels (NaN hotel_booking_rate): {test_df['hotel_booking_rate'].isna().sum()}")

    rows_before = len(test_df)
    test_df = test_df.merge(dest_performance, on='srch_destination_id', how='left') # dropped later
    print(f"AFTER MERGE dest_performance into test: {rows_before} -> {len(test_df)} rows (diff: {len(test_df) - rows_before})")

    # Replace NaN with destination mean, then global mean
    global_booking_rate = train_df['booking_bool'].mean()
    global_click_rate = train_df['click_bool'].mean()
    global_position_avg = train_df['position'].mean()

    test_df['hotel_booking_rate'] = test_df['hotel_booking_rate'].fillna(test_df['dest_booking_rate'])
    test_df['hotel_booking_rate'] = test_df['hotel_booking_rate'].fillna(global_booking_rate)

    test_df['hotel_click_rate'] = test_df['hotel_click_rate'].fillna(test_df['dest_click_rate'])
    test_df['hotel_click_rate'] = test_df['hotel_click_rate'].fillna(global_click_rate)

    test_df['hotel_avg_position'] = test_df['hotel_avg_position'].fillna(global_position_avg)
    test_df['hotel_n_appearances'] = test_df['hotel_n_appearances'].fillna(0)

    # Drop the destination columns
    test_df = test_df.drop(columns=['dest_booking_rate', 'dest_click_rate'])
    train_df = train_df.drop(columns=['dest_booking_rate', 'dest_click_rate'])

    # Merge new features into training df

    print(f"\nFINAL TEST SHAPE: {test_df.shape}")
    print(f"FINAL TEST unique srch_ids: {test_df['srch_id'].nunique()}")

    return train_df, test_df

def validate_hotel_performance(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    '''
    Validates the output of extract_hotel_performance_train and extract_hotel_performance_test.
    Checks for correct columns, no unexpected NaNs, and sensible value ranges.
    '''

    print("=" * 50)
    print("VALIDATING HOTEL PERFORMANCE FEATURES")
    print("=" * 50)

    expected_cols = ['hotel_booking_rate', 'hotel_click_rate', 'hotel_avg_position', 'hotel_n_appearances']

    # Columns check
    for col in expected_cols:
        assert col in train_df.columns, f"Missing {col} in train_df"
        assert col in test_df.columns, f"Missing {col} in test_df"
    print("✓ All expected columns present in both train and test")

    # NaN checks
    for col in expected_cols:
        train_nans = train_df[col].isna().sum()
        test_nans = test_df[col].isna().sum()
        assert train_nans == 0, f"{col} has {train_nans} NaNs in train_df"
        assert test_nans == 0, f"{col} has {test_nans} NaNs in test_df"
    print("✓ No NaNs in any hotel performance columns")

    # Value range checks
    for df, name in [(train_df, 'train'), (test_df, 'test')]:
        assert df['hotel_booking_rate'].between(0, 1).all(), f"hotel_booking_rate out of [0,1] in {name}"
        assert df['hotel_click_rate'].between(0, 1).all(), f"hotel_click_rate out of [0,1] in {name}"
        assert (df['hotel_n_appearances'] >= 0).all(), f"hotel_n_appearances negative in {name}"
        assert (df['hotel_avg_position'] > 0).all(), f"hotel_avg_position <= 0 in {name}"
    print("✓ All values within expected ranges")

    # Check: Rates should be low (most hotels are not booked) 
    print(f"\nTrain hotel_booking_rate stats:\n{train_df['hotel_booking_rate'].describe().round(4)}")
    print(f"\nTest hotel_booking_rate stats:\n{test_df['hotel_booking_rate'].describe().round(4)}")

    # Check: train and test distributions should be similar
    for col in expected_cols:
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        diff = abs(train_mean - test_mean)
        print(f"\n{col}: train mean={train_mean:.4f}, test mean={test_mean:.4f}, diff={diff:.4f}")
        threshold = 1.0 if col == 'hotel_n_appearances' else 0.1
        assert diff < threshold, f"Large distribution mismatch in {col} between train and test: {diff:.4f}"
    print("\n ✓ Train and test distributions are similar")

    # Check for new hotels in test
    new_hotels = test_df['hotel_n_appearances'].eq(0).sum()
    print(f"\nNew hotels in test (unseen in train): {new_hotels} ({100*new_hotels/len(test_df):.2f}%)")

    print("\n ✓ All validation checks passed!")


# train, test = extract_hotel_performance_test(train_df, test_df)

# print(test.head(20))
# validate_hotel_performance(train, test)