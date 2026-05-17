import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def create_interaction_matrix(df):
    # Map bookings to 5, clicks to 1, others to 0
    conditions = [df['booking_bool'] == 1, df['click_bool'] == 1]
    choices = [5, 1]
    df = df.copy()
    df['relevance'] = np.select(conditions, choices, default=0)
    
    agg_df = df.groupby(['srch_destination_id', 'prop_id'])['relevance'].max().reset_index()
    
    # Convert IDs to category codes for contiguous sparse indexing
    dest_ids = agg_df['srch_destination_id'].astype('category')
    prop_ids = agg_df['prop_id'].astype('category')
    
    row_indices = dest_ids.cat.codes
    col_indices = prop_ids.cat.codes
    
    dest_id_map = dict(enumerate(dest_ids.cat.categories))
    prop_id_map = dict(enumerate(prop_ids.cat.categories))
    
    matrix = csr_matrix(
        (agg_df['relevance'], (row_indices, col_indices)),
        shape=(len(dest_ids.cat.categories), len(prop_ids.cat.categories))
    )
    return matrix, dest_id_map, prop_id_map

def get_svd_embeddings(matrix, dest_id_map, prop_id_map, n_components=20):
    n_components = min(n_components, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    dest_features = svd.fit_transform(matrix)
    hotel_features = svd.components_.T
    
    # Generate context column names
    user_cols = []
    for i in range(n_components):
        user_cols.append(f'svd_user_{i}')
        
    dest_df = pd.DataFrame(dest_features, columns=user_cols)
    dest_df['srch_destination_id'] = dest_df.index.map(dest_id_map)
    dest_df = dest_df[['srch_destination_id'] + user_cols]
    
    # Generate hotel column names
    hotel_cols = []
    for i in range(n_components):
        hotel_cols.append(f'svd_hotel_{i}')
        
    hotel_df = pd.DataFrame(hotel_features, columns=hotel_cols)
    hotel_df['prop_id'] = hotel_df.index.map(prop_id_map)
    hotel_df = hotel_df[['prop_id'] + hotel_cols]
    
    return dest_df, hotel_df

def compute_dot_product_features(df, n_components=20):
    user_cols = []
    for i in range(n_components):
        user_cols.append(f'svd_user_{i}')
        
    hotel_cols = []
    for i in range(n_components):
        hotel_cols.append(f'svd_hotel_{i}')
        
    u_vals = df[user_cols].values
    h_vals = df[hotel_cols].values
    
    # Matrix-multiply to capture reconstruction relevance
    df['svd_dot_product'] = (u_vals * h_vals).sum(axis=1)
    
    # Vectorized assignment to prevent DataFrame fragmentation warnings
    interact_matrix = u_vals * h_vals
    interact_cols = {}
    for i in range(n_components):
        interact_cols[f'svd_interact_{i}'] = interact_matrix[:, i]
        
    df = df.assign(**interact_cols)
    return df

def run_svd_pipeline_user(train_df, test_df, n_components=20, add_dot_product=True):
    # Destination mapping fixes the test set cold-start/zero-out bug
    matrix, dest_id_map, prop_id_map = create_interaction_matrix(train_df)
    dest_features_df, hotel_features_df = get_svd_embeddings(matrix, dest_id_map, prop_id_map, n_components)
    
    # Merge context profiles and item factors
    train_df = train_df.merge(dest_features_df, on='srch_destination_id', how='left')
    test_df = test_df.merge(dest_features_df, on='srch_destination_id', how='left')
    
    train_df = train_df.merge(hotel_features_df, on='prop_id', how='left')
    test_df = test_df.merge(hotel_features_df, on='prop_id', how='left')
    
    user_cols = []
    for i in range(n_components):
        user_cols.append(f'svd_user_{i}')
        
    hotel_cols = []
    for i in range(n_components):
        hotel_cols.append(f'svd_hotel_{i}')
        
    # Fill unseen test entries securely with zero vectors
    train_df[user_cols] = train_df[user_cols].fillna(0.0)
    train_df[hotel_cols] = train_df[hotel_cols].fillna(0.0)
    test_df[user_cols] = test_df[user_cols].fillna(0.0)
    test_df[hotel_cols] = test_df[hotel_cols].fillna(0.0)
    
    if add_dot_product:
        train_df = compute_dot_product_features(train_df, n_components)
        test_df = compute_dot_product_features(test_df, n_components)
        
    return train_df, test_df

if __name__ == "__main__":
    print("Pipeline loaded.")