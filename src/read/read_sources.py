import pandas as pd
from ..lib import lib

class ReadSources:

    root: "Root"
    def __init__(self, root: "Root") -> None:
        self.root = root
    
    @property
    def dataframe(self) -> pd.DataFrame:
        if self._data is None:
            path = self.root.config.read.data_sources["transactions"]
            self._data = lib.load_csv(path)

        return self._data
