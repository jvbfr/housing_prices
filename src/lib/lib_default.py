from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize data in the dataframe."""
    return (data - data.mean()) / data.std()

def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(filepath)