from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection, metrics

def create_plot(data: Any, plot_type: str = "line") -> plt.Figure:
    """Create a plot based on data."""
    fig, ax = plt.subplots()
    if plot_type == "line":
        ax.plot(data)
    elif plot_type == "bar":
        ax.bar(range(len(data)), data)
    return fig