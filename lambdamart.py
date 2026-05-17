#
# NOTE WE NEED hotel_performance.py AND other_features.py FOR THIS TO RUN !!!
# USES lightlbm 
#

import pandas as pd
from pandas import Series
from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from hotel_performance import extract_hotel_performance_train, extract_hotel_performance_test, extract_hotel_revenue_features
from other_features import add_search_relative_features, add_basic_features, add_user_cluster_features_with_validation, cap_price_usd, aggregate_competitor_rates
from collaborativefiltering import run_svd_pipeline
import matplotlib.pyplot as plt
import lightgbm as lgb

TESTING_MODE = False

print("starting...")

# DEBUG
raw_test = pd.read_csv('test_set_VU_DM.csv', low_memory=False)

# Open training set
if not TESTING_MODE:
    train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)
else:
    # Use only 10% of data while testing
    train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False, nrows=500000)

# position is a string and we don't want that !
train_df['position'] = pd.to_numeric(train_df['position'], errors='coerce')
assert 'gross_bookings_usd' in train_df.columns

# open test set
if not TESTING_MODE:
    test_df = pd.read_csv('test_set_VU_DM.csv', low_memory=False)
else:
    # Use only 10% of data while testing
    test_df = pd.read_csv('test_set_VU_DM.csv', low_memory=False, nrows=5000)

# split data
splitter = GroupShuffleSplit(
    test_size=0.2,
    random_state=42
)
# split on search_id
train_idx, valid_idx = next(
    splitter.split(train_df, groups=train_df['srch_id'])
)

# training and validation set !
train_part = train_df.iloc[train_idx]
valid_part = train_df.iloc[valid_idx]

# feature engineer
train_fold, _, _ = extract_hotel_performance_train(train_part)

_, valid_fold = extract_hotel_performance_test(
    train_part,
    valid_part
)


print("feature engineering")
# feature engineer test set
_, test_fold = extract_hotel_performance_test(train_df, test_df)

train_fold, valid_fold = extract_hotel_revenue_features(train_fold, valid_fold)
_, test_fold = extract_hotel_revenue_features(train_df, test_fold)
# cap prices
# train_fold, test_fold = cap_price_usd(train_fold, test_fold)
# _, valid_fold = cap_price_usd(train_fold, valid_fold)

# Add basic features
train_fold = add_basic_features(train_fold)
valid_fold = add_basic_features(valid_fold)
test_fold = add_basic_features(test_fold)

# ADD SEARCH RELATIVE FEATURES
train_fold = add_search_relative_features(train_fold)
valid_fold = add_search_relative_features(valid_fold)
test_fold = add_search_relative_features(test_fold)

# aggregation of competitor data
train_fold = aggregate_competitor_rates(train_fold)
valid_fold = aggregate_competitor_rates(valid_fold)
test_fold = aggregate_competitor_rates(test_fold)

# add cluster features
train_fold, valid_fold, test_fold = add_user_cluster_features_with_validation(train_fold, valid_fold, test_fold)

# collaborative filtering
# train_fold, test_fold = run_svd_pipeline(train_fold, test_fold, 20)
# _, valid_fold = run_svd_pipeline(train_fold, valid_fold, 20, validation=True)

# add relevance
train_fold['relevance'] = 0
train_fold.loc[train_fold['click_bool'] == 1, 'relevance'] = 1
train_fold.loc[train_fold['booking_bool'] == 1, 'relevance'] = 5

valid_fold['relevance'] = 0
valid_fold.loc[valid_fold['click_bool'] == 1, 'relevance'] = 1
valid_fold.loc[valid_fold['booking_bool'] == 1, 'relevance'] = 5

# write something good here
# exclude: booking_bool, click_bool, relevance and gross_booking_usd
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
    # 'dest_avg_revenue',
    # 'dest_std_revenue',
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

train_fold = train_fold.sort_values('srch_id')
valid_fold = valid_fold.sort_values('srch_id')

# for ranking
train_group = train_fold.groupby('srch_id').size().to_numpy()
valid_group = valid_fold.groupby('srch_id').size().to_numpy()

# matrices
X_train = train_fold[features]
y_train = train_fold['relevance']

X_valid = valid_fold[features]
y_valid = valid_fold['relevance']

# training the lambda
# Final model from optuna
model = lgb.LGBMRanker(
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

print("training model")
# fit model
model.fit(
    X_train,
    y_train,
    group=train_group,
    eval_set=[(X_valid, y_valid)],
    eval_group=[valid_group],
        callbacks=[
        lgb.log_evaluation(period=100), 
        lgb.early_stopping(stopping_rounds=50)
    ]
)

# get validation prediction
valid_fold['prediction'] = model.predict(X_valid)

# predict on test
X_test = test_fold[features]
test_fold['prediction'] = model.predict(X_test)


# submission to csv
submission = (
    test_fold
    .sort_values(['srch_id', 'prediction'], ascending=[True, False])
    [['srch_id', 'prop_id']]
)

# save to csv
submission.to_csv('submission/fake-group154_submission1.csv', index=False)

# validation to csv
validation = (
    valid_fold
    .sort_values(['srch_id', 'prediction'], ascending=[True, False])
    [['srch_id', 'prop_id']]
)
validation.to_csv('submission/group154_validation2.csv', index=False)


# check features
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(importance)

print("==== DEBUG ======")
print(f"Number of trees: {model.n_iter_}")
print(f"X_train shape: {X_train.shape}")
print(f"X_train all zeros: {(X_train == 0).all().all()}")
print(f"X_train NaN count: {X_train.isna().sum().sum()}")
print(f"Features in model: {len(features)}")
print(f"Columns in X_train: {X_train.shape[1]}")
print('===== NANS ========')
nan_by_feature = X_train.isna().sum().sort_values(ascending=False)
print(nan_by_feature[nan_by_feature > 0])

# only remove features with exactly zero importance
zero_importance = importance[importance == 0].index.tolist()
print('===== ZERO IMPORTANCE: =====', features)

print('===== NEW IMPORTANCE =====')
importance = pd.Series(
    model.booster_.feature_importance(importance_type='gain'),
    index=features
).sort_values(ascending=False)
print(importance)

importance.plot(kind='bar', figsize=(12, 5), title='Feature Importances')
plt.tight_layout()
plt.savefig('figures/feature_importance.png')
plt.show()

# plot training curve
results = model.evals_result_
ndcg_scores = results['valid_0']['ndcg@5']

plt.figure(figsize=(10, 5))
plt.plot(ndcg_scores, label='validation NDCG@5')
plt.xlabel('Tree number')
plt.ylabel('NDCG@5')
plt.title('Training curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('figures/training_curve.png')
plt.show()


# are there any NaNs in the submission?
print('\nnans in submission: ', submission.isna().sum())

# how many unique searches?
print(f"\nUnique searches in submission: {submission['srch_id'].nunique()}")
print(f"Unique searches in test: {test_df['srch_id'].nunique()}")

# how many unique searches?
print(f"Unique searches in submission: {submission['srch_id'].nunique()}")
print(f"Unique searches in test: {test_df['srch_id'].nunique()}")
