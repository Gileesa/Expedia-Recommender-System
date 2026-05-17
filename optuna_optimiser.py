#
# Optuna hyperparameter and feature optimiser for LambdaMART
# NOTE WE NEED hotel_performance.py AND other_features.py FOR THIS TO RUN !!!
#

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from hotel_performance import extract_hotel_performance_train, extract_hotel_performance_test
from other_features import add_search_relative_features, add_basic_features, only_train_test_add_user_cluster_features, cap_price_usd, aggregate_competitor_rates
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')

# load data
train_df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)
test_df = pd.read_csv('test_set_VU_DM.csv', low_memory=False)
train_df['position'] = pd.to_numeric(train_df['position'], errors='coerce') # since was string first


print("Running feature engineering...")

# split data first: validation and training
splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, valid_idx = next(splitter.split(train_df, groups=train_df['srch_id']))
train_part = train_df.iloc[train_idx]
valid_part = train_df.iloc[valid_idx]

# feature engineer train
train_fold, _, _ = extract_hotel_performance_train(train_part)
train_fold, _ = cap_price_usd(train_fold, valid_part.copy())
train_fold = add_basic_features(train_fold)
train_fold = add_search_relative_features(train_fold)
train_fold = aggregate_competitor_rates(train_fold)

# feature engineer valid
_, valid_fold = extract_hotel_performance_test(train_part, valid_part)
_, valid_fold = cap_price_usd(train_fold.copy(), valid_fold)
valid_fold = add_basic_features(valid_fold)
valid_fold = add_search_relative_features(valid_fold)
valid_fold = aggregate_competitor_rates(valid_fold)

# relevance scores
train_fold['relevance'] = 0
train_fold.loc[train_fold['click_bool'] == 1, 'relevance'] = 1
train_fold.loc[train_fold['booking_bool'] == 1, 'relevance'] = 5

valid_fold['relevance'] = 0
valid_fold.loc[valid_fold['click_bool'] == 1, 'relevance'] = 1
valid_fold.loc[valid_fold['booking_bool'] == 1, 'relevance'] = 5

# sort per search_id
train_fold = train_fold.sort_values('srch_id')
valid_fold = valid_fold.sort_values('srch_id')

train_group = train_fold.groupby('srch_id').size().to_numpy()
valid_group = valid_fold.groupby('srch_id').size().to_numpy()

print("Feature engineering done!")

# define features
# features that we know work will always be included
# other features are optional and chosen through optuna
ALL_FEATURES = {
    # core hotel properties (always include)
    'core': [
        'price_usd', 'prop_starrating', 'prop_review_score',
        'prop_location_score1', 'prop_location_score2',
        'prop_log_historical_price', 'prop_brand_bool', 'promotion_flag',
    ],
    # hotel performance (always include)
    'hotel_performance': [
        'hotel_booking_rate', 'hotel_click_rate',
        'hotel_avg_position', 'hotel_n_appearances',
    ],
    # search relative (always include)
    'search_relative': [
        'price_pct_rank', 'price_usd_diff', 'price_usd_zscore',
        'price_per_night', 'price_per_person',
        'prop_starrating_diff', 'prop_starrating_zscore',
        'prop_review_score_diff', 'prop_review_score_zscore',
    ],
    # competitor aggregates (optional)
    'competitor': [
        'comp_n_available', 'comp_n_cheaper', 'comp_n_more_expensive',
        'comp_n_same', 'comp_rate_mean', 'comp_expedia_wins', 'comp_win_rate',
    ],
    # search context (optional)
    'search_context': [
        'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
        'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
        'log_booking_win', 'log_length_stay', 'random_bool',
    ],
    # user history (optional)
    'user_history': [
        'visitor_location_country_id', 'srch_query_affinity_score',
        'orig_destination_distance', 'star_pref_delta', 'same_country',
    ],
    # basic engineered (optional)
    'basic_engineered': [
        'log_price', 'quality_score', 'prop_country_id',
        'search_month', 'search_day', 'search_hour',
        'total_people', 'is_family', 'is_solo', 'is_couple',
        'is_group', 'people_per_room', 'is_long_stay',
        'is_last_minute', 'is_planned',
    ],
}

# Optuna objective
def objective(trial: optuna.Trial) -> float:

    # Hyperparams for LambdaMART
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'verbosity': -1,
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 2.0),
    }

    # Feature selection
    # core and hotel_performance always included
    features = ALL_FEATURES['core'] + ALL_FEATURES['hotel_performance'] + ALL_FEATURES['search_relative']

    # optional feature groups
    for group_name in ['competitor', 'search_context', 'user_history', 'basic_engineered']:
        if trial.suggest_categorical(f'use_{group_name}', [True, False]):
            features += ALL_FEATURES[group_name]

    # check all features exist
    missing = [f for f in features if f not in train_fold.columns]
    if missing:
        return 0.0

    # Train 
    # let's first see most negative values
    X_train = train_fold[features]
    print('=== LOWEST VALUES IN TRAIN ===')
    print(X_train.min().sort_values().head(10))
    print("=" * 50)

    # filling NaNs is always done after feature engineering to avoid weird calculations
    X_train = train_fold[features].fillna(-999) # -999 is far outside range, so model can learn this is 'weird'
    y_train = train_fold['relevance']
    X_valid = valid_fold[features].fillna(-999)
    y_valid = valid_fold['relevance']

    # define model
    model = lgb.LGBMRanker(**params)
    model.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_valid, y_valid)],
        eval_group=[valid_group],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )

    # Eval
    score = model.best_score_['valid_0']['ndcg@5']
    return score


# Run optimisation
print("Starting Optuna optimisation...")

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)

study.optimize(
    objective,
    n_trials=50,          # can also decrease to process faster
    show_progress_bar=True
)

# Results
print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)
print(f"\nBest NDCG@5: {study.best_value:.4f}")
print(f"Best trial: {study.best_trial.number}")
print(f"\nBest hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

# printing the selected feature groups
print(f"\nSelected feature groups:")
for group_name in ['competitor', 'search_context', 'user_history', 'basic_engineered']:
    selected = study.best_trial.params.get(f'use_{group_name}', False)
    print(f"  {group_name}: {'✓' if selected else '✗'}")

# save results
results_df = study.trials_dataframe()
results_df.to_csv('optuna_results.csv', index=False)
print(f"\n[INFO] Results saved to optuna_results.csv")

# ── Visualisation ─────────────────────────────────────────────
try:
    import plotly
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html('optuna_history.html')
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html('optuna_param_importance.html')
    print("[INFO] Visualisations saved to optuna_history.html and optuna_param_importance.html")
except ImportError:
    print("[WARNING] Could not create visualisations: No module named 'plotly'")