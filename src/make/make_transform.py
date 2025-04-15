import numpy as np
import pandas as pd


class MakeTransform:
    root: "Root"

    def __init__(self, root: "Root") -> None:
        self.root = root
    
    def transform(self) -> pd.DataFrame:
        """Transform the processed data."""
        transformed = data.copy()

        for col in transformed.columns:
            if transformed[col].dtype == np.number:
                transformed[f"{col}_squared"] = transformed[col] ** 2
        
        self._transformed_data = transformed

        return transformed