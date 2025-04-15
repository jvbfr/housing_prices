import pandas as pd

from ..lib import lib


class ReadOrigins:

    root: "Root"
    def __init__(self, root: "Root") -> None:
        self.root = root
    
    @property
    def dataframe(self) -> pd.DataFrame:

        path = self.root.config.read.data_sources["housing"]
        return lib.load_csv(path)