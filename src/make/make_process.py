import pandas as pd

from ..lib import lib


class MakeProcess:
    root: "Root"

    def __init__(self, root: "Root") -> None:
        self.root = root
    
    @property
    def dataframe(self) -> pd.DataFrame:
        """Process the input dataframe."""
        processed = data.copy()
        # Apply some processing based on config parameters
        feature_cols = self._config.make.feature_columns
        processed = processed[feature_cols]
        processed = lib.normalize_data(processed)
        self._processed_data = processed

        return processed