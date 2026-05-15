#
#
#

import pandas as pd
import math as m
import matplotlib.pyplot as plt

df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)

def print_nan_percentage():
    nan_counts = df.isnull().sum()
    nan_pct = df.isnull().mean() * 100

    summary = pd.DataFrame({'NaN Count': nan_counts, 'NaN %': nan_pct.round(2)})
    summary = summary[summary['NaN Count'] > 0].sort_values('NaN %', ascending=False)

    print(summary.to_string())



def plot_histograms():
    features = [
    'prop_starrating',
    'prop_review_score',
    'price_usd',
    'prop_log_historical_price',
    'promotion_flag',
    'prop_location_score1',
    'prop_location_score2',
    'visitor_hist_starrating',
    'visitor_hist_adr_usd',
    'srch_query_affinity_score',
    'orig_destination_distance',
    'comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate',
    'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate',
    ]

    # filter for desired columns
    features = [f for f in features if f in df.columns]

    n_cols = 3
    n_rows = m.ceil(len(features) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(features):
        data = df[col].dropna()
        
        # Cap extreme outliers at 99th percentile for readability
        # cap = data.quantile(0.99)
        # data = data[data <= cap]
        
        axes[i].hist(data, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
        axes[i].grid(axis='y', alpha=0.3)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Histograms of Strong Predictive Features', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('feature_histograms.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_outliers(col='price_usd'):
    ''' prints 99th quantile'''
    p99 = df[col].quantile(0.99)

    outliers = df[df[col] > p99][col]

    print(f"99th percentile threshold: {p99:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"\nOutlier statistics:")
    print(outliers.describe())
    print(f"\nTop 20 highest values:")
    print(outliers.sort_values(ascending=False).head(20).to_string())
    print("Conclusion: cap price at 99th-percentile.")
    print("Don't delete but create flag for model to know it was capped")

# plot_histograms()
# test
print_outliers()

#
#
#

import pandas as pd
import math as m
import matplotlib.pyplot as plt

df = pd.read_csv('training_set_VU_DM.csv', low_memory=False)

def print_nan_percentage():
    nan_counts = df.isnull().sum()
    nan_pct = df.isnull().mean() * 100

    summary = pd.DataFrame({'NaN Count': nan_counts, 'NaN %': nan_pct.round(2)})
    summary = summary[summary['NaN Count'] > 0].sort_values('NaN %', ascending=False)

    print(summary.to_string())



def plot_histograms():
    features = [
    'prop_starrating',
    'prop_review_score',
    'price_usd',
    'prop_log_historical_price',
    'promotion_flag',
    'prop_location_score1',
    'prop_location_score2',
    'visitor_hist_starrating',
    'visitor_hist_adr_usd',
    'srch_query_affinity_score',
    'orig_destination_distance',
    'comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate',
    'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate',
    ]

    # filter for desired columns
    features = [f for f in features if f in df.columns]

    n_cols = 3
    n_rows = m.ceil(len(features) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(features):
        data = df[col].dropna()
        
        # Cap extreme outliers at 99th percentile for readability
        # cap = data.quantile(0.99)
        # data = data[data <= cap]
        
        axes[i].hist(data, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
        axes[i].grid(axis='y', alpha=0.3)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Histograms of Strong Predictive Features', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('feature_histograms.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_outliers(col='price_usd'):
    ''' prints 99th quantile'''
    p99 = df[col].quantile(0.99)

    outliers = df[df[col] > p99][col]

    print(f"99th percentile threshold: {p99:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"\nOutlier statistics:")
    print(outliers.describe())
    print(f"\nTop 20 highest values:")
    print(outliers.sort_values(ascending=False).head(20).to_string())
    print("Conclusion: cap price at 99th-percentile.")
    print("Don't delete but create flag for model to know it was capped")

# plot_histograms()
# test
print_outliers()
print_nan_percentage()
