from typing import Any, Dict, List, Optional

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
        
        from dbdisplay import (
            enable_databricks_display,  # !pip install databricks-dbdisplay
        )
        enable_databricks_display()  
    except:
        pass  

    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    sns.set()