from typing import Dict


class ConfigMake:

    root: "Root"
    def __init__(self, root: "Root") -> None:
        self.root = root
    
    @property
    def data_sources(self) -> Dict[str, str]:

        return {
        }
    
    @property
    def base_path(self) -> str:
        return {
        }