
#
#
#

import pandas as pd
import numpy as np

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
    # we take the log; let's remove bc leakage maybe 
    # if 'position' in df.columns:
    #     df['log_position'] = np.log1p(df['position'])

    return df



# FEATURE ENGINEERING

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Date ────────────────────────────────────────────────
    df["date_time"]       = pd.to_datetime(df["date_time"])
    df["search_month"]    = df["date_time"].dt.month
    df["search_day"]      = df["date_time"].dt.dayofweek
    df["search_hour"]     = df["date_time"].dt.hour         

    # ── Travel party ────────────────────────────────────────
    df["total_people"]    = df["srch_adults_count"] + df["srch_children_count"]
    df["is_family"]       = (df["srch_children_count"] > 0).astype(int)
    df["is_solo"]         = (df["srch_adults_count"] == 1) & (df["srch_children_count"] == 0)
    df["is_solo"]         = df["is_solo"].astype(int)
    df["is_couple"]       = (df["srch_adults_count"] == 2) & (df["srch_children_count"] == 0)
    df["is_couple"]       = df["is_couple"].astype(int)
    df["is_group"]        = (df["srch_adults_count"] > 2).astype(int)
    df["people_per_room"] = df["total_people"] / df["srch_room_count"].replace(0, 1)

    # ── Trip style ───────────────────────────────────────────
    df["is_long_stay"]    = (df["srch_length_of_stay"] >= 4).astype(int)
    df["is_weekend_trip"] = df["srch_saturday_night_bool"].astype(int)
    df["is_last_minute"]  = (df["srch_booking_window"] <= 3).astype(int)
    df["is_planned"]      = (df["srch_booking_window"] > 14).astype(int)
    df["log_booking_win"] = np.log1p(df["srch_booking_window"])
    df["log_length_stay"] = np.log1p(df["srch_length_of_stay"])

    # ── Visitor history ──────────────────────────────────────
    df["has_hist_star"]       = df["visitor_hist_starrating"].notnull().astype(int)
    df["has_hist_price"]      = df["visitor_hist_adr_usd"].notnull().astype(int)
    df["is_high_end_user"]    = (df["visitor_hist_starrating"] >= 4).astype(int)
    df["star_pref_delta"]     = (
        df["prop_starrating"] - df["visitor_hist_starrating"]
    )  # positive = above their usual standard
    df["price_pref_delta"]    = (
        df["price_usd"] - df["visitor_hist_adr_usd"]
    )

    # ── Geographic ──────────────────────────────────────────
    df["same_country"]        = (
        df["visitor_location_country_id"] == df["prop_country_id"]
    ).astype(int)

    # ── Price ────────────────────────────────────────────────
    df["log_price"]           = np.log1p(df["price_usd"])

    # ── Property quality ─────────────────────────────────────
    df["quality_score"]       = (
        df["prop_starrating"] * 0.4 + df["prop_review_score"] * 0.6
    )  # weighted blend used as a single quality axis
    df["has_promotion"]       = df["promotion_flag"].fillna(0).astype(int)

    # ── Expedia positioning signals ──────────────────────────
    df["log_position"]        = np.log1p(df.get("position", 0))
    # position is only available in training; harmless NaN in test

    # ── Per-query relative ranks (competitive context) ───────
    for col, ascending in [
        ("price_usd",         True),
        ("prop_starrating",   False),
        ("prop_review_score", False),
        ("prop_location_score1", False),
    ]:
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("srch_id")[col].rank(
                ascending=ascending, method="average"
            )

    return df


def add_property_agg_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Compute historical booking/click rates per property from training data.
    Avoids leakage by computing on the full training set before the split.

    I will be skipping this function because I fear leakage problems
    """
    # Click-through and booking rates per property
    # has no LOO so probably leakage !!!
    prop_stats = (
        train.groupby("prop_id")
        .agg(
            prop_click_rate  = ("click_bool",   "mean"),
            prop_book_rate   = ("booking_bool", "mean"),
            prop_n_queries   = ("srch_id",      "nunique"),
            prop_mean_price  = ("price_usd",    "mean"),
            prop_mean_star   = ("prop_starrating", "mean"),
            prop_mean_review = ("prop_review_score", "mean"),
        )
        .reset_index()
    )

    # Country-level booking rates
    country_stats = (
        train.groupby("prop_country_id")
        .agg(
            country_book_rate = ("booking_bool", "mean"),
            country_click_rate = ("click_bool",  "mean"),
        )
        .reset_index()
    )

    train = train.merge(prop_stats,    on="prop_id",         how="left")
    train = train.merge(country_stats, on="prop_country_id", how="left")

    test  = test.merge(prop_stats,    on="prop_id",         how="left")
    test  = test.merge(country_stats, on="prop_country_id", how="left")

    # Fill unseen properties (cold start) with a conservative low value
    for df in [train, test]:
        for col in ["prop_click_rate", "prop_book_rate"]:
            df[col] = df[col].fillna(df[col].quantile(0.25))
        for col in ["country_book_rate", "country_click_rate"]:
            df[col] = df[col].fillna(df[col].median())

    return train, test


def only_train_test_add_user_cluster_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple:
    """
    K-Means user segmentation on search-level features.
    Fit on combined train+test so test users get proper assignments.

    Creating clusters based partly on test set seems like it might cause a leakage problem
    """
    train = train.copy()
    test  = test.copy()

    train["_is_train"] = 1
    test["_is_train"]  = 0
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    user_features = [
        "srch_length_of_stay", "srch_booking_window", "srch_room_count",
        "is_family", "is_solo", "is_couple", "is_group",
        "total_people", "people_per_room",
        "is_long_stay", "is_weekend_trip", "is_last_minute", "is_planned",
        "has_hist_star", "is_high_end_user", "same_country",
        "search_month", "search_day", "search_hour",
    ]

    user_df = combined[["srch_id"] + user_features].drop_duplicates("srch_id").copy()

    log_cols = [
        "srch_booking_window", "srch_length_of_stay", "total_people", "people_per_room"
    ]
    for col in log_cols:
        user_df[col] = np.log1p(user_df[col])

    user_df[user_features] = (
        user_df[user_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-999)
    )

    X_scaled = StandardScaler().fit_transform(user_df[user_features])

    N_CLUSTERS = 6   # one extra: "family planner" now distinct from "long-stay planner"
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    user_df["user_cluster"] = km.fit_predict(X_scaled)

    combined = combined.merge(
        user_df[["srch_id", "user_cluster"]], on="srch_id", how="left"
    )

    # One-hot encode clusters for the model
    for k in range(N_CLUSTERS):
        combined[f"cluster_{k}"] = (combined["user_cluster"] == k).astype(int)

    train_out = combined[combined["_is_train"] == 1].drop(columns=["_is_train"])
    test_out  = combined[combined["_is_train"] == 0].drop(columns=["_is_train"])
    return train_out, test_out




def add_user_cluster_features_with_validation(
    train: pd.DataFrame, 
    valid: pd.DataFrame,
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    K-Means user segmentation on search-level features.
    Fit on train only, apply to valid and test separately to avoid leakage.
    """
    train = train.copy()
    valid = valid.copy()
    test = test.copy()

    user_features = [
        "srch_length_of_stay", "srch_booking_window", "srch_room_count",
        "is_family", "is_solo", "is_couple", "is_group",
        "total_people", "people_per_room",
        "is_long_stay", "is_weekend_trip", "is_last_minute", "is_planned",
        "has_hist_star", "is_high_end_user", "same_country",
        "search_month", "search_day", "search_hour",
    ]

    def get_user_df(df):
        user_df = df[["srch_id"] + user_features].drop_duplicates("srch_id").copy()
        log_cols = ["srch_booking_window", "srch_length_of_stay", "total_people", "people_per_room"]
        for col in log_cols:
            user_df[col] = np.log1p(user_df[col])
        user_df[user_features] = (
            user_df[user_features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-999)
        )
        return user_df

    # fit scaler and kmeans on train only
    train_user_df = get_user_df(train)
    scaler = StandardScaler()
    # fit model and uses zscore for it (=scaling)
    X_train_scaled = scaler.fit_transform(train_user_df[user_features])

    N_CLUSTERS = 6
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    train_user_df["user_cluster"] = km.fit_predict(X_train_scaled)

    # apply to valid and test using fitted scaler and kmeans
    # note we do NOT fit on the validation and test set to avoid leakage
    # apply kmeans to validation set
    valid_user_df = get_user_df(valid)
    X_valid_scaled = scaler.transform(valid_user_df[user_features]) # no fitting, only scaling
    valid_user_df["user_cluster"] = km.predict(X_valid_scaled)

    # apply kmeans to test set
    test_user_df = get_user_df(test)
    X_test_scaled = scaler.transform(test_user_df[user_features])  # no fitting, only scaling
    test_user_df["user_cluster"] = km.predict(X_test_scaled)

    # merge back
    for df, user_df in [(train, train_user_df), (valid, valid_user_df), (test, test_user_df)]:
        df_merged = df.merge(user_df[["srch_id", "user_cluster"]], on="srch_id", how="left")
        for k in range(N_CLUSTERS):
            df_merged[f"cluster_{k}"] = (df_merged["user_cluster"] == k).astype(int)
    
    # reassign since merge creates new df
    train = train.merge(train_user_df[["srch_id", "user_cluster"]], on="srch_id", how="left")
    valid = valid.merge(valid_user_df[["srch_id", "user_cluster"]], on="srch_id", how="left")
    test = test.merge(test_user_df[["srch_id", "user_cluster"]], on="srch_id", how="left")

    for df in [train, valid, test]:
        for k in range(N_CLUSTERS):
            df[f"cluster_{k}"] = (df["user_cluster"] == k).astype(int)

    return train, valid, test


def cap_price_usd(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Caps price_usd at the 99th percentile to avoid infeasible prices.
    Computes cap from training data to avoid leakage

    Params:
        train_df (pd.DataFrame): the training pandas dataframe.
        test_df (pd.DataFrame): the test pandas dataframe.
    Returns:
        train_df (pd.DataFrame): training dataframe with capped price_usd.
        test_df (pd.DataFrame): test dataframe with capped price_usd.
    """
    # compute cap from training data only
    p99 = train_df['price_usd'].quantile(0.99)
    print(f"price_usd 99th percentile cap: {p99:.2f}")
    print(f"Rows capped in train: {(train_df['price_usd'] > p99).sum()}")
    print(f"Rows capped in test: {(test_df['price_usd'] > p99).sum()}")

    # add flag before capping
    train_df['price_usd_was_capped'] = (train_df['price_usd'] > p99).astype(int)
    test_df['price_usd_was_capped'] = (test_df['price_usd'] > p99).astype(int)

    # cap price_usd
    train_df['price_usd'] = train_df['price_usd'].clip(upper=p99)
    test_df['price_usd'] = test_df['price_usd'].clip(upper=p99)

    return train_df, test_df

def aggregate_competitor_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates competitor rate columns (comp1_rate to comp8_rate) into
    meaningful summary features, handling NaN values appropriately.

    NaN means no competitive data available for that competitor.
    Values are: +1 if Expedia is cheaper, 0 if same, -1 if Expedia is more expensive.

    Args:
        df (pd.DataFrame): input dataframe
    Returns:
        df (pd.DataFrame): dataframe with aggregated competitor features
    """
    comp_rate_cols = [f'comp{i}_rate' for i in range(1, 9)]

    # number of competitors with data for this hotel/search
    df['comp_n_available'] = df[comp_rate_cols].notna().sum(axis=1)

    # number of competitors Expedia is cheaper than
    df['comp_n_cheaper'] = (df[comp_rate_cols] == 1).sum(axis=1)

    # number of competitors Expedia is more expensive than
    df['comp_n_more_expensive'] = (df[comp_rate_cols] == -1).sum(axis=1)

    # number of competitors have the same price
    df['comp_n_same'] = (df[comp_rate_cols] == 0).sum(axis=1)

    # net competitive advantage: positive means Expedia is cheaper overall
    df['comp_rate_mean'] = df[comp_rate_cols].mean(axis=1)

    # is Expedia cheaper than majority of competitors?
    df['comp_expedia_wins'] = (df['comp_n_cheaper'] > df['comp_n_more_expensive']).astype(int)

    # fraction of available competitors Expedia beats
    df['comp_win_rate'] = df['comp_n_cheaper'] / df['comp_n_available'].replace(0, np.nan)
    df['comp_win_rate'] = df['comp_win_rate'].fillna(0)

    return df