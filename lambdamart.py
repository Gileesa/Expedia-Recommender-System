#
#
#

import pandas as pd
from pandas import Series
from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from hotel_performance import extract_hotel_performance_train, extract_hotel_performance_test
import matplotlib.pyplot as plt

TESTING_MODE = False

# Open training set
if not TESTING_MODE:
    train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)
else:
    # Use only 10% of data while testing
    train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False, nrows=500000)

# position is a string and we don't want that !
train_df['position'] = pd.to_numeric(train_df['position'], errors='coerce')

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

# DEBUG
print(train_part.columns.tolist())
print('booking_bool' in train_part.columns)

# feature engineer
train_fold, _, _ = extract_hotel_performance_train(train_part)

valid_fold, _ = extract_hotel_performance_test(
    train_part,
    valid_part
)

# feature engineer test set
test_fold, _ = extract_hotel_performance_test(train_df, test_df)

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
    # original 9
    'price_usd',
    'prop_starrating',
    'prop_review_score',
    'prop_location_score1',
    'prop_location_score2',
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
model = XGBRanker(
    objective='rank:ndcg',
    eval_metric='ndcg@5',

    learning_rate=0.05,
    max_depth=6,
    n_estimators=300,

    subsample=0.8,
    colsample_bytree=0.8,

    early_stopping_rounds=50,
    random_state=42
)

# fit model
model.fit(
    X_train,
    y_train,
    group=train_group,
    eval_set=[(X_valid, y_valid)],
    eval_group=[valid_group],
    verbose=True,
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
submission.to_csv('submission/group154_submission1.csv', index=False)

# validation to csv
validation = (
    valid_fold
    .sort_values(['srch_id', 'prediction'], ascending=[True, False])
    [['srch_id', 'prop_id']]
)
validation.to_csv('submission/group154_validation1.csv', index=False)


# check features
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(importance)

importance.plot(kind='bar', figsize=(12, 5), title='Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()


# DEBUG
print("=== prop_location_score2 ===")
print(f"dtype: {train_df['prop_location_score2'].dtype}")
print(f"null count: {train_df['prop_location_score2'].isna().sum()}")
print(f"null %: {train_df['prop_location_score2'].isna().mean() * 100:.2f}%")
print(f"\nValue stats:")
print(train_df['prop_location_score2'].describe())
print(f"\nSample of non-null values:")
print(train_df['prop_location_score2'].dropna().head(20).tolist())
print(f"\nAre there any 0 values?")
print(f"Count of 0s: {(train_df['prop_location_score2'] == 0).sum()}")
print(f"\nDistribution of nulls vs booking_bool:")
print(train_df.groupby(train_df['prop_location_score2'].isna())['booking_bool'].mean())