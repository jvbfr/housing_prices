from typing import Dict


class ConfigRead:

    root: "Root"
    def __init__(self, root: "Root") -> None:
        self.root = root
    
    @property
    def data_sources(self) -> Dict[str, str]:

        return {
            "housing": "../data/housing.csv",
            "housing_resampled": "../data/housing_resampled.csv"
        }
    
    @property
    def base_path(self) -> str: 
        return {

        }
