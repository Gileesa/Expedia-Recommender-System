import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple, Dict
from sklearn.decomposition import TruncatedSVD


def create_interaction_matrix(df: pd.DataFrame) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    Creates a user-hotel interaction matrix from search data.
    Matrix shape: (n_searches, n_hotels)
    
    Args:
        df with columns 'srch_id', 'prop_id', 'booking_bool', 'click_bool'.
        
    Returns:
        tuple with:
            - sparse interaction matrix (csr_matrix)
            - mapping from row indices to srch_id
            - mapping from column indices to prop_id
    """
    
    # 1. Calculate Relevance Scores
    # Booking is stronger signal than click
    conditions = [
        (df['booking_bool'] == 1),
        (df['click_bool'] == 1)
    ]
    choices = [5, 1]
    relevance = np.select(conditions, choices, default=0)
    
    # 2. Map IDs to contiguous integers (0 to N-1)
    srch_ids = df['srch_id'].astype('category')
    prop_ids = df['prop_id'].astype('category')
    
    row_indices = srch_ids.cat.codes
    col_indices = prop_ids.cat.codes
    
    # Store mappings for later use
    srch_id_map = dict(enumerate(srch_ids.cat.categories))
    prop_id_map = dict(enumerate(prop_ids.cat.categories))
    
    # 3. Build Sparse Matrix
    matrix = csr_matrix(
        (relevance, (row_indices, col_indices)),
        shape=(len(srch_ids.cat.categories), len(prop_ids.cat.categories))
    )
    
    return matrix, srch_id_map, prop_id_map


def get_svd_embeddings(interaction_matrix: csr_matrix, 
                       srch_id_map: Dict[int, int],
                       prop_id_map: Dict[int, int],
                       n_components: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs SINGLE SVD decomposition to extract both user and hotel embeddings.
    
    Matrix = U x sum x V^T
    - U: Per-search latent factors (what kind of hotel each search wants)
    - V^T: Per-hotel latent factors (what characteristics each hotel has)
    
    Args:
        interaction_matrix: Sparse matrix of shape (n_searches, n_hotels)
        srch_id_map: Mapping from matrix indices to original srch_id
        prop_id_map: Mapping from matrix indices to original prop_id
        n_components: Number of latent dimensions
        
    Returns:
        Tuple of (user_features_df, hotel_features_df)
    """
    
    # Single SVD decomposition
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    # U × sum (user/search embeddings with singular values absorbed)
    user_features = svd.fit_transform(interaction_matrix)  # shape: (n_searches, k)
    
    # V^T transposed to V (hotel embeddings)
    hotel_features = svd.components_.T  # shape: (n_hotels, k)
    
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
    
    # Create user DataFrame
    user_cols = [f'svd_user_{i}' for i in range(n_components)]
    user_df = pd.DataFrame(user_features, columns=user_cols)
    user_df['srch_id'] = user_df.index.map(srch_id_map)
    user_df = user_df[['srch_id'] + user_cols]
    
    print(f"SVD user features shape: {user_df.shape}")
    
    # Create hotel DataFrame
    hotel_cols = [f'svd_hotel_{i}' for i in range(n_components)]
    hotel_df = pd.DataFrame(hotel_features, columns=hotel_cols)
    hotel_df['prop_id'] = hotel_df.index.map(prop_id_map)
    hotel_df = hotel_df[['prop_id'] + hotel_cols]
    
    print(f"SVD hotel features shape: {hotel_df.shape}")
    
    return user_df, hotel_df


def compute_dot_product_features(df: pd.DataFrame, n_components: int = 20) -> pd.DataFrame:
    """
    Computes dot product between user and hotel embeddings for each search-hotel pair.
    This creates interaction features for LambdaMART.
    
    The dot product reconstructs the predicted relevance from the SVD model:
    relevance = user_embedding dotproduct hotel_embedding
    
    Args:
        df: DataFrame with both svd_user_* and svd_hotel_* columns
        n_components: Number of SVD components used
        
    Returns:
        DataFrame with additional dot product features
    """
    
    user_cols = [f'svd_user_{i}' for i in range(n_components)]
    hotel_cols = [f'svd_hotel_{i}' for i in range(n_components)]
    
    # Dot product: predicted relevance from SVD
    df['svd_dot_product'] = (df[user_cols].values * df[hotel_cols].values).sum(axis=1)

    for i in range(n_components):
        df[f'svd_interact_{i}'] = df[f'svd_user_{i}'] * df[f'svd_hotel_{i}']
    
    print(f"Added SVD dot product features")
    
    return df


def run_svd_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     n_components: int = 20, 
                     add_dot_product: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete SVD pipeline: extracts both user and hotel features from a SINGLE decomposition,
    merges them, and optionally computes dot products for LambdaMART.
    
    Args:
        train_df: Training data with columns including srch_id, prop_id, booking_bool, click_bool
        test_df: Test data with same columns
        n_components: Number of latent factors
        add_dot_product: Whether to compute user-hotel dot products
        
    Returns:
        (train_df_enriched, test_df_enriched) with SVD features
    """
    
    print("Building interaction matrix from training data")
    interaction_matrix, srch_id_map, prop_id_map = create_interaction_matrix(train_df)
    
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"Matrix density: {interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.4%}")
    
    # SINGLE SVD decomposition for both user and hotel features
    print("\nPerforming SVD decomposition...")
    user_features_df, hotel_features_df = get_svd_embeddings(
        interaction_matrix, 
        srch_id_map, 
        prop_id_map, 
        n_components
    )
    
    # Merge user features
    print("\nMerging user features...")
    train_df = train_df.merge(user_features_df, on='srch_id', how='left')
    test_df = test_df.merge(user_features_df, on='srch_id', how='left')
    
    # Merge hotel features
    print("Merging hotel features...")
    train_df = train_df.merge(hotel_features_df, on='prop_id', how='left')
    test_df = test_df.merge(hotel_features_df, on='prop_id', how='left')
    
    # Handle cold start (new searches or hotels in test set)
    user_cols = [f'svd_user_{i}' for i in range(n_components)]
    hotel_cols = [f'svd_hotel_{i}' for i in range(n_components)]
    
    print("\nHandling cold start...")
    test_df[user_cols] = test_df[user_cols].fillna(0.0)
    test_df[hotel_cols] = test_df[hotel_cols].fillna(0.0)
    
    # Compute dot product features for LambdaMART
    if add_dot_product:
        print("\nComputing dot product features...")
        train_df = compute_dot_product_features(train_df, n_components)
        test_df = compute_dot_product_features(test_df, n_components)
    
    print("\n SVD Pipeline completed successfully!")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df


if __name__ == "__main__":
    print("Functions loaded successfully.")