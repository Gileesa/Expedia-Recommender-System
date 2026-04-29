#
# Code where we generate features for the hotel's performance
# mostly based on data that is only available for the test set
#


# We are trying to predict booking_bool -> make sure no leakage!

import pandas as pd


# Open training set
train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)

# open test set
test_df = pd.read_csv('test_set_VU_DM.csv', low_memory=False)

def extract_hotel_performance(train_df, test_df):
    '''
    Function that extracts hotel performance features from training data
    to merge into test data based on prop_id.
    If a new prop_id appears in the test data, inpute the destination mean or 
    the global mean and add flas_is_new_hotel.

    Params:
    - train_df (pd.DataFrame): the training pandas dataframe
    - test_df (pd.DataFrame): the test pandas dataframe
    Returns:
    - test_df (pd.DataFrame): the test dataframe with new features
    '''

    # Compute features from training data only
    hotel_performance= train_df.groupby('prop_id').agg(
        hotel_booking_rate=('booking_bool', 'mean'), # did it get booked 0/1
        hotel_click_rate=('click_bool', 'mean'), # did it get clicked 0/1
        hotel_avg_position=('position', 'mean'),  # how highly Expedia usually ranks it
        hotel_n_appearances=('prop_id', 'count') # how often hotel appears in dataset
    ).reset_index()


    # Get destination mean
    dest_stats = train_df.groupby('srch_destination_id').agg(
        dest_booking_rate=('booking_bool', 'mean'),
        dest_click_rate=('click_bool', 'mean')
    ).reset_index()

    # Merge into test set
    test_df = test_df.merge(hotel_performance, on='prop_id', how='left')
    test_df = test_df.merge(dest_stats, on='srch_destination_id', how='left')

    # Add flag for new hotels
    test_df['flag_is_new_hotel'] = test_df['hotel_booking_rate'].isna().astype(int)

    # Replace NaN with destination mean, then global mean
    global_booking_rate = train_df['booking_bool'].mean()
    global_click_rate = train_df['click_bool'].mean()

    test_df['hotel_booking_rate'] = test_df['hotel_booking_rate'].fillna(test_df['dest_booking_rate'])
    test_df['hotel_booking_rate'] = test_df['hotel_booking_rate'].fillna(global_booking_rate)

    test_df['hotel_click_rate'] = test_df['hotel_click_rate'].fillna(test_df['dest_click_rate'])
    test_df['hotel_click_rate'] = test_df['hotel_click_rate'].fillna(global_click_rate)

    # Drop the  destination columns
    test_df = test_df.drop(columns=['dest_booking_rate', 'dest_click_rate'])

    return test_df

print(test_df.head(20))