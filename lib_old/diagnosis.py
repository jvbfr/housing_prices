import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def missing_values(df):
    missing_abs = df.isnull().sum().rename('absolute missing').reset_index()
    missing_abs.columns = ['column', 'absolute missing']
    
    missing_abs['% missing'] = (missing_abs['absolute missing'] / len(df)) * 100
    missing_abs['% missing'] = missing_abs['% missing'].round(2)
    
    result = missing_abs[['column', '% missing', 'absolute missing']]
    
    return result

def summary_num(df):
    num_df = df.select_dtypes(include=[np.number])
    
    if num_df.empty:
        return pd.DataFrame()
    
    stats = {
        'min': num_df.min(),
        '25%': num_df.quantile(0.25),
        'median': num_df.median(),
        '75%': num_df.quantile(0.75),
        'max': num_df.max(),
        'range': num_df.max() - num_df.min(),
        'mean': num_df.mean(),
        'std': num_df.std(),
        'skew': num_df.skew(),
        'kurtosis': num_df.kurtosis()
    }
    
    result = pd.DataFrame(stats).reset_index()
    result.columns = [
        'attribute', 'min', '25%', 'median', '75%', 'max',
        'range', 'mean', 'std', 'skew', 'kurtosis'
    ]
    
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(2)
    
    return result


def summary_cat(df):
    cat_df = df.select_dtypes(include=['object', 'category'])
    
    if cat_df.empty:
        return pd.DataFrame()
    
    stats = {
        'count': cat_df.count(),
        'unique': cat_df.nunique(),
        'top': cat_df.mode().iloc[0],
        'freq': cat_df.apply(lambda x: x.value_counts().iloc[0]),
        'missing': df.isnull().sum()[cat_df.columns]
    }
    
    result = pd.DataFrame(stats).reset_index()
    result.columns = [
        'attribute', 'count', 'unique', 'most_frequent',
        'frequency', 'missing'
    ]
    
    result['missing_pct'] = (result['missing'] / len(df)).round(4) * 100
    result['frequency_pct'] = (result['frequency'] / len(df)).round(4) * 100
    
    result = result[[
        'attribute', 'count', 'unique', 'most_frequent',
        'frequency', 'frequency_pct', 'missing', 'missing_pct'
    ]]
    
    return result

# Analisys of distribution per category with the target
def cat_feature_analisys(cat_col, target, figsize=(12, 6)):
    """
    Analyzes the relationship between a categorical column and a target column by computing
    statistics and plotting KDE distributions for each category.
    
    Parameters:
    cat_col (pd.Series): Categorical column with different groups.
    target (pd.Series): Target numerical column to analyze.
    figsize (tuple): Figure size for the plot, default is (12, 6).
    """
    stats_df = target.groupby(cat_col).agg(['mean', 'median', 'std', 'max', 'min'])
    
    for category in stats_df.index:
        print(f"Statistics for category '{category}':")
        print(f"  Mean: {stats_df.loc[category, 'mean']:.2f}")
        print(f"  Median: {stats_df.loc[category, 'median']:.2f}")
        print(f"  Standard Deviation: {stats_df.loc[category, 'std']:.2f}")
        print(f"  Max: {stats_df.loc[category, 'max']:.2f}")
        print(f"  Min: {stats_df.loc[category, 'min']:.2f}\n")
    
    plt.figure(figsize=figsize)
    for category in stats_df.index:
        sns.kdeplot(target[cat_col == category], label=f'{category}', common_norm= True, fill = True)
    plt.xlabel('Target Value')
    plt.ylabel('Density')
    plt.title('KDE Plot of Target Distribution by Category')
    plt.legend(title=cat_col.name)
    plt.show()