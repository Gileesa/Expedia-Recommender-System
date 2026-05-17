#
# NOTE WE NEED hotel_performance.py AND other_features.py FOR THIS TO RUN !!!
# USES xgboost NOT lightlbm SO THAT MIGHT CHANGE PERFORMANCE
#

import pandas as pd
from pandas import Series
# from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from hotel_performance import extract_hotel_performance_train, extract_hotel_performance_test,  extract_hotel_revenue_features
from collaborativefiltering import run_svd_pipeline
from other_features import add_search_relative_features, add_basic_features, only_train_test_add_user_cluster_features, cap_price_usd, aggregate_competitor_rates
import matplotlib.pyplot as plt
import lightgbm as lgb


train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)
test_df = pd.read_csv('test_set_VU_DM.csv', low_memory=False)

# position is a string and we don't want that !
train_df['position'] = pd.to_numeric(train_df['position'], errors='coerce')

assert 'gross_bookings_usd' in train_df.columns


# train on full training data
train_full, _, _ = extract_hotel_performance_train(train_df)
_, test_fold = extract_hotel_performance_test(train_df, test_df)
train_full, test_fold = extract_hotel_revenue_features(train_full, test_fold)

# cap price
train_full, test_fold = cap_price_usd(train_full, test_fold)

# add features to train
train_full = add_basic_features(train_full)
train_full = add_search_relative_features(train_full)

# add features to test
test_fold = add_basic_features(test_fold)
test_fold = add_search_relative_features(test_fold)

train_full, test_fold = only_train_test_add_user_cluster_features(train_full, test_fold)

# aggregation of competitor data
train_full = aggregate_competitor_rates(train_full)
test_fold = aggregate_competitor_rates(test_fold)

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
# features = [
#     # original 9
#     'price_usd',
#     'prop_starrating',
#     'prop_review_score',
#     'prop_location_score1',
#     'prop_location_score2',

#     # HOTEL PERFORMANCE
#     'hotel_booking_rate',
#     'hotel_click_rate',
#     'hotel_avg_position',
#     'hotel_n_appearances',

#     # search context
#     'srch_length_of_stay',
#     'srch_booking_window',
#     'srch_adults_count',
#     'srch_children_count',
#     'srch_room_count',
#     'srch_saturday_night_bool',
    
#     # hotel properties
#     'prop_brand_bool',
#     'prop_log_historical_price',
#     'promotion_flag',
#     'prop_country_id',

#     # user history
#     # 'visitor_hist_starrating',
#     # 'visitor_hist_adr_usd',
#     'visitor_location_country_id',

#     # search/hotel match
#     'srch_query_affinity_score',
#     'orig_destination_distance',

#     # competitor data
#     # 'comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate',
#     # 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate',

#     # for debiasing
#     'random_bool',

#     # search-relative; note no log_position !
#     'price_pct_rank',
#     'price_usd_diff',
#     'price_usd_zscore',
#     'price_per_night',
#     'price_per_person',
#     'prop_starrating_diff',
#     'prop_starrating_zscore',
#     'prop_review_score_diff',
#     'prop_review_score_zscore',

#     # add_basic_features
#     # 'search_month',
#     # 'search_day',
#     # 'search_hour',
#     # 'total_people',
#     # 'is_family',
#     # 'is_solo',
#     # 'is_couple',
#     # 'is_group',
#     # 'people_per_room',
#     # 'is_long_stay',
#     # 'is_last_minute',
#     # 'is_planned',
#     'log_booking_win',
#     'log_length_stay',
#     # 'has_hist_star',
#     # 'has_hist_price',
#     # 'is_high_end_user',
#     'star_pref_delta',
#     # 'price_pref_delta',
#     'same_country',
#     'log_price',
#     'quality_score',

#     # add_user_cluster_features
#     # 'cluster_0', 'cluster_1', 'cluster_2',
#     # 'cluster_3', 'cluster_4', 'cluster_5',

#     # 'svd_feature_0', 'svd_feature_1', 'svd_feature_2', 'svd_feature_3',
#     # 'svd_feature_4', 'svd_feature_5', 'svd_feature_6',
#     # 'svd_feature_7', 'svd_feature_8', 'svd_feature_9', 'svd_feature_10',
#     # 'svd_feature_11', 'svd_feature_12', 'svd_feature_13', 'svd_feature_14',
#     # 'svd_feature_15', 'svd_feature_16', 'svd_feature_17', 'svd_feature_18',
#     # 'svd_feature_19'

#     'comp_n_available',
#     'comp_n_cheaper',
#     'comp_n_more_expensive', 
#     'comp_n_same',
#     'comp_rate_mean',
#     'comp_expedia_wins',
#     'comp_win_rate'
# ]

features = [
    # core hotel properties
    'price_usd',
    'prop_starrating',
    'prop_review_score',
    'prop_location_score1',
    'prop_location_score2',
    'prop_log_historical_price',
    'prop_brand_bool',
    'promotion_flag',

    # hotel performance
    'hotel_booking_rate',
    'hotel_click_rate',
    'hotel_avg_position',
    'hotel_n_appearances',
    'hotel_avg_gross_usd',
    # 'hotel_revenue_zscore',
    # 'hotel_total_gross_usd'

    # search relative
    'price_pct_rank',
    'price_usd_diff',
    'price_usd_zscore',
    'price_per_night',
    'price_per_person',
    'prop_starrating_diff',
    'prop_starrating_zscore',
    'prop_review_score_diff',
    'prop_review_score_zscore',

    # competitor (use_competitor: True)
    'comp_n_available',
    'comp_n_cheaper',
    'comp_n_more_expensive',
    'comp_n_same',
    'comp_rate_mean',
    'comp_expedia_wins',
    'comp_win_rate',

    # user history (use_user_history: True)
    'visitor_location_country_id',
    'srch_query_affinity_score',
    'orig_destination_distance',
    'star_pref_delta',
    'same_country',

    # basic engineered (use_basic_engineered: True)
    'log_price',
    'quality_score',
    'prop_country_id',
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
]


X_train_full = train_full[features]
y_train_full = train_full['relevance']

# Final model from optuna
model_final = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    ndcg_eval_at=[5],
    learning_rate=0.015361234354331139,
    max_depth=7,
    num_leaves=229,
    n_estimators=1244,
    subsample=0.8208868378919796,
    colsample_bytree=0.6160492884083315,
    min_child_samples=72,
    reg_alpha=9.80921883242501e-06,
    reg_lambda=0.6184965914240641,
    min_gain_to_split=1.8791682372762821,
    random_state=42,
    verbosity=-1
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
submission.to_csv('submission/group154_submission13.csv', index=False)
