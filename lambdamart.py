#
#
#

import pandas as pd
from pandas import Series
from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from hotel_performance import extract_hotel_performance_train, extract_hotel_performance_test

# Open training set
train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)

# open test set
test_df = pd.read_csv('test_set_VU_DM.csv', low_memory=False)

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
train_fold, hotel_performance = extract_hotel_performance_train(train_part)

valid_fold, _ = extract_hotel_performance_test(
    train_part,
    valid_part
)

# add relevance
train_fold['relevance'] = 0
train_fold.loc[train_df['click_bool'] == 1, 'relevance'] = 1
train_fold.loc[train_df['booking_bool'] == 1, 'relevance'] = 5

valid_fold['relevance'] = 0
valid_fold.loc[train_df['click_bool'] == 1, 'relevance'] = 1
valid_fold.loc[train_df['booking_bool'] == 1, 'relevance'] = 5

# write something good here
# exclude: booking_bool, click_bool, relevance and gross_booking_usd
features = [
    'price_usd',
    'prop_starrating',
    'prop_review_score',
    'prop_location_score1',
    'prop_location_score2',
    'hotel_booking_rate',
    'hotel_click_rate',
    'hotel_avg_position',
    'hotel_n_appearances'
]

train_df = train_df.sort_values('srch_id')
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

    random_state=42
)

# fit model
model.fit(
    X_train,
    y_train,
    group=train_group,
    eval_set=[(X_valid, y_valid)],
    eval_group=[valid_group],
    eval_metric='ndcg@5',
    verbose=True,
    early_stopping_rounds=50
)

# get test prediction
valid_fold['prediction'] = model.predict(X_valid)

# submission to csv
submission = (
    test_df
    .sort_values(['srch_id', 'prediction'], ascending=[True, False])
    [['srch_id', 'prop_id']]
)
submission.to_csv('submission/submission1.csv', index=False)