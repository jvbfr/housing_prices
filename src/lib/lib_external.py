from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from sklearn import metrics, model_selection, preprocessing


def create_plot(data: Any, plot_type: str = "line") -> plt.Figure:
    """Create a plot based on data."""
    fig, ax = plt.subplots()
    if plot_type == "line":
        ax.plot(data)
    elif plot_type == "bar":
        ax.bar(range(len(data)), data)
    return fig

def jupyter_settings():
    try:
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        display(HTML('<style>.container { width:90% !important; }</style>'))
    
    except:
        pass  

    pd.set_option('display.float_format', '{:.2f}'.format)
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    sns.set()

def oversample_minority_categories(df, cat_col, target_proportion=None, random_state=None):
    """
    Oversamples minority categories in a categorical column to balance the dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    cat_col (str): Name of the categorical column to balance.
    target_proportion (dict or None): Desired proportion for each category. If None, the smallest category is oversampled to match the next smallest.
    random_state (int or None): Random seed for reproducibility.
    
    Returns:
    pd.DataFrame: Dataframe with oversampled minority categories.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate current counts and proportions
    counts = df[cat_col].value_counts()
    proportions = counts / counts.sum()
    
    print("Original proportions:")
    print(proportions)
    
    # Determine target counts
    if target_proportion is None:
        # Default: oversample the smallest category to match the next smallest
        minority_cat = proportions.idxmin()
        next_smallest_prop = proportions.nsmallest(2).iloc[1]
        target_counts = counts.copy()
        target_counts[minority_cat] = int(counts[minority_cat] * (next_smallest_prop / proportions[minority_cat]))
    else:
        # Use provided target proportions
        total = len(df)
        target_counts = {cat: int(total * prop) for cat, prop in target_proportion.items()}
        target_counts = pd.Series(target_counts)
    
    print("\nTarget counts:")
    print(target_counts)
    
    # Oversample each minority category
    oversampled_dfs = []
    for category in counts.index:
        category_df = df[df[cat_col] == category]
        current_count = len(category_df)
        target_count = target_counts[category]
        
        if target_count > current_count:
            # Calculate how many additional samples are needed
            additional_samples = target_count - current_count
            # Randomly sample with replacement
            oversampled_samples = category_df.sample(n=additional_samples, replace=True, random_state=random_state)
            # Combine original and oversampled
            category_df = pd.concat([category_df, oversampled_samples])
        
        oversampled_dfs.append(category_df)
    
    # Combine all categories and shuffle
    balanced_df = pd.concat(oversampled_dfs)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Verify new proportions
    new_counts = balanced_df[cat_col].value_counts()
    new_proportions = new_counts / new_counts.sum()
    
    print("\nNew proportions after oversampling:")
    print(new_proportions)
    
    return balanced_df