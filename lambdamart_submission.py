#
# NOTE WE NEED hotel_performance.py AND other_features.py FOR THIS TO RUN !!!
# USES xgboost NOT lightlbm SO THAT MIGHT CHANGE PERFORMANCE
#

import pandas as pd
from pandas import Series
# from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from hotel_performance import extract_hotel_performance_train, extract_hotel_performance_test
from collaborativefiltering import run_svd_pipeline
from other_features import add_search_relative_features, add_basic_features, only_train_test_add_user_cluster_features, cap_price_usd
import matplotlib.pyplot as plt
import lightgbm as lgb


train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)
test_df = pd.read_csv('test_set_VU_DM.csv', low_memory=False)

# position is a string and we don't want that !
train_df['position'] = pd.to_numeric(train_df['position'], errors='coerce')



# train on full training data
train_full, _, _ = extract_hotel_performance_train(train_df)
# train_full = train_df
train_full = add_basic_features(train_full)
train_full = add_search_relative_features(train_full)

# feature engineer test set
_, test_fold = extract_hotel_performance_test(train_df, test_df)
# test_fold = test_df
test_fold = add_basic_features(test_fold)
test_fold = add_search_relative_features(test_fold)

# cap price
train_full, test_fold = cap_price_usd(train_full, test_fold)

train_full, test_fold = only_train_test_add_user_cluster_features(train_full, test_fold)

# collaborative filtering
train_full, test_fold = run_svd_pipeline(train_full, test_fold, 20, n_components=20, add_dot_product=True)

# adding relevance
train_full['relevance'] = 0
train_full.loc[train_full['click_bool'] == 1, 'relevance'] = 1
train_full.loc[train_full['booking_bool'] == 1, 'relevance'] = 5

# sort per search id
train_full = train_full.sort_values('srch_id')
train_group_full = train_full.groupby('srch_id').size().to_numpy()

test_fold = test_fold.sort_values('srch_id')

# choose features
# exclude: booking_bool, click_bool, relevance and gross_booking_usd
features = [
    # original 9
    'price_usd',
    'prop_starrating',
    'prop_review_score',
    'prop_location_score1',
    'prop_location_score2',

    # HOTEL PERFORMANCE
    'hotel_booking_rate',
    'hotel_click_rate',
    'hotel_avg_position',
    'hotel_n_appearances',

    # search context
    'srch_length_of_stay',
    'srch_booking_window',
    'srch_adults_count',
    'srch_children_count',
    'srch_room_count',
    'srch_saturday_night_bool',
    
    # hotel properties
    'prop_brand_bool',
    'prop_log_historical_price',
    'promotion_flag',
    'prop_country_id',

    # user history
    'visitor_hist_starrating',
    'visitor_hist_adr_usd',
    'visitor_location_country_id',

    # search/hotel match
    'srch_query_affinity_score',
    'orig_destination_distance',

    # competitor data
    'comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate',
    'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate',

    # for debiasing
    'random_bool',

    # search-relative; note no log_position !
    'price_pct_rank',
    'price_usd_diff',
    'price_usd_zscore',
    'price_per_night',
    'price_per_person',
    'prop_starrating_diff',
    'prop_starrating_zscore',
    'prop_review_score_diff',
    'prop_review_score_zscore',

    # add_basic_features
    'search_month',
    'search_day',
    'search_hour',
    'total_people',
    'is_family',
    'is_solo',
    'is_couple',
    'is_group',
    'people_per_room',
    'is_long_stay',
    'is_last_minute',
    'is_planned',
    'log_booking_win',
    'log_length_stay',
    # 'has_hist_star',
    # 'has_hist_price',
    'is_high_end_user',
    'star_pref_delta',
    'price_pref_delta',
    'same_country',
    'log_price',
    'quality_score',

    # add_user_cluster_features
    # 'cluster_0', 'cluster_1', 'cluster_2',
    # 'cluster_3', 'cluster_4', 'cluster_5',

    # SVD FEATURES
    # User embeddings (what kind of hotel the search wants)
    'svd_user_0', 'svd_user_1', 'svd_user_2', 'svd_user_3',
    'svd_user_4', 'svd_user_5', 'svd_user_6', 'svd_user_7',
    'svd_user_8', 'svd_user_9', 'svd_user_10', 'svd_user_11',
    'svd_user_12', 'svd_user_13', 'svd_user_14', 'svd_user_15',
    'svd_user_16', 'svd_user_17', 'svd_user_18', 'svd_user_19',
    
    # Hotel embeddings (what the hotel offers)
    'svd_hotel_0', 'svd_hotel_1', 'svd_hotel_2', 'svd_hotel_3',
    'svd_hotel_4', 'svd_hotel_5', 'svd_hotel_6', 'svd_hotel_7',
    'svd_hotel_8', 'svd_hotel_9', 'svd_hotel_10', 'svd_hotel_11',
    'svd_hotel_12', 'svd_hotel_13', 'svd_hotel_14', 'svd_hotel_15',
    'svd_hotel_16', 'svd_hotel_17', 'svd_hotel_18', 'svd_hotel_19',
    
    # Dot product (predicted relevance from collaborative filtering)
    'svd_dot_product',
    
    # Individual interactions (per-dimension products)
    'svd_interact_0', 'svd_interact_1', 'svd_interact_2', 'svd_interact_3',
    'svd_interact_4', 'svd_interact_5', 'svd_interact_6', 'svd_interact_7',
    'svd_interact_8', 'svd_interact_9', 'svd_interact_10', 'svd_interact_11',
    'svd_interact_12', 'svd_interact_13', 'svd_interact_14', 'svd_interact_15',
    'svd_interact_16', 'svd_interact_17', 'svd_interact_18', 'svd_interact_19',
]

X_train_full = train_full[features]
y_train_full = train_full['relevance']

# note: no eval_set since we have no validation set anymore
model_final = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    ndcg_eval_at=[5],
    learning_rate=0.05,
    max_depth=6,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    # early_stopping_rounds=50,
    random_state=42
)

model_final.fit(
    X_train_full,
    y_train_full,
    group=train_group_full,
)

# predict on test
X_test = test_fold[features]
test_fold['prediction'] = model_final.predict(X_test)


# submission to csv
submission = (
    test_fold
    .sort_values(['srch_id', 'prediction'], ascending=[True, False])
    [['srch_id', 'prop_id']]
)

# sanity checks
print(f"Submission shape: {submission.shape}")
print(f"Unique srch_ids in submission: {submission['srch_id'].nunique()}")
print(f"Unique srch_ids in test: {test_df['srch_id'].nunique()}")
print(f"NaNs in submission: {submission.isna().sum().sum()}")
print(submission.head(10))

# save to csv
submission.to_csv('submission/group154_submission8.csv', index=False)
