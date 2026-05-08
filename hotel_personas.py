import pandas as pd

def load_data(file_path='training_set_VU_DM.csv'):
    return pd.read_csv(file_path, low_memory=False)

def get_hotel_properties(df):
    threshold = df["prop_location_score2"].quantile(0.8)

    df["is_budget_hotel"] = (df["prop_starrating"] <= 2)
    df["is_posh_hotel"] = (df["prop_starrating"] >= 4) & (df["prop_review_score"] >= 4.0)  
    df["is_mid_scale_hotel"] = (df["prop_starrating"] == 3)
    df["is_boutique_hotel"] = (df["prop_brand_bool"] == 0) & (df["prop_review_score"] >= 4.5)
    df["is_well_located_hotel"] = df["prop_location_score2"] > threshold
    df["is_discounted_hotel"] = (df["promotion_flag"] == 1)
    
    return df[["is_budget_hotel", "is_posh_hotel", "is_mid_scale_hotel", "is_boutique_hotel", "is_well_located_hotel", "is_discounted_hotel"]]


def get_hotel_data(df):
    data_hotels = [
"prop_country_id",
"prop_id",
"prop_starrating",
"prop_review_score",
"prop_brand_bool",
"is_budget_hotel",
"is_posh_hotel",
"is_mid_scale_hotel",
"is_boutique_hotel",
"is_well_located_hotel",
"price_usd",
"promotion_flag",
"orig_destination_distance"
 ] 

    df_hotel_profiles = df[data_hotels].drop_duplicates(subset="prop_id")
    return df_hotel_profiles

