import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple, Dict
from sklearn.decomposition import TruncatedSVD


def create_interaction_matrix(df: pd.DataFrame) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    transforms the input DataFrame into a sparse interaction matrix suitable for collaborative filtering models (like SVD).
    
    Args:
        df with columns 'srch_id', 'prop_id', 'booking_bool', 'click_bool'.
        
    Returns:
        tuple with - sparse interaction matrix (csr_matrix)
              - mapping from row indices to srch_id
              - mapping from column indices to prop_id
    """
    
    # 1. calculate Relevance Scores
    # booked = 5, clicked = 1. np.select is here much faster than .apply()
    conditions = [
        (df['booking_bool'] == 1),
        (df['click_bool'] == 1)
    ]
    choices = [5, 1]
    relevance = np.select(conditions, choices, default=0)
    
    # 2. Map IDs to unique integers (0 to N-1)
    # This is necessary for the sparse matrix indexing, as IDs may have gaps
    srch_ids = df['srch_id'].astype('category')
    prop_ids = df['prop_id'].astype('category')
    
    row_indices = srch_ids.cat.codes
    col_indices = prop_ids.cat.codes
    
    # stores the mapping from the original IDs to the new integer indices, which is crucial for interpreting the model's output later on
    srch_id_map = dict(enumerate(srch_ids.cat.categories))
    prop_id_map = dict(enumerate(prop_ids.cat.categories))
    
    # 3. Build the Sparse Matrix (Compressed Sparse Row format)
    matrix = csr_matrix(
        (relevance, (row_indices, col_indices)),
        shape=(len(srch_ids.cat.categories), len(prop_ids.cat.categories))
    )
    
    return matrix, srch_id_map, prop_id_map
    


def get_svd_hotel_features(interaction_matrix: csr_matrix, prop_id_map: Dict[int, int], n_components: int = 20) -> np.ndarray:
    """
    Applies Singular Value Decomposition (SVD) to the interaction matrix to extract latent features for hotels.
    
    Args:
        interaction_matrix: Sparse matrix of shape (n_users, n_hotels) with relevance scores.
        n_components: Number of latent features to extract.
        
    Returns:
        A dense matrix of shape (n_hotels, n_components) containing the hotel features.
    """
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(interaction_matrix)

    hotel_features = svd.components_.T  # Transpose to get hotels as rows

    col_names = [f'svd_feature_{i}' for i in range(n_components)]

    df_svd = pd.DataFrame(hotel_features, columns=col_names)

    df_svd['prop_id'] = df_svd.index.map(prop_id_map)  # Map back to original prop_id
           
    cols = ['prop_id'] + col_names
    df_svd = df_svd[cols]

    print(f"SVD hotel features shape: {df_svd.shape}")

    return df_svd

def run_svd_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, n_components: int = 20, validation:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the entire SVD pipeline: creates interaction matrix, extracts hotel features, 
    and merges them into both the train and test sets.
    
    Args:
        train_df: DataFrame containing the training data.
        test_df: DataFrame containing the test data.
        n_components: Number of latent features to extract (default 20).
        
    Returns:
        Tuple containing the updated (train_df, test_df) with SVD features.
    """
    
    # Build the interaction matrix from the training data
    interaction_matrix, srch_id_map, prop_id_map = create_interaction_matrix(train_df)
    
    # Get SVD hotel features
    hotel_features_df = get_svd_hotel_features(interaction_matrix, prop_id_map, n_components)
    
    # Merge the SVD features into the training set
    if not validation:
        train_df = train_df.merge(hotel_features_df, on='prop_id', how='left')

    #Merge the SVD features into the test set
    test_df = test_df.merge(hotel_features_df, on='prop_id', how='left')

    # This one fixes our cold start problem
    svd_columns = [col for col in train_df.columns if 'svd_feature_' in col]
    test_df[svd_columns] = test_df[svd_columns].fillna(0.0)

    print("SVD Pipeline finished.")
    
    return train_df, test_df

if __name__ == "__main__":
    print("Functions loaded successfully.")