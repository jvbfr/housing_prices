from typing import Annotated
from .read_origins import ReadOrigins
from .read_sources import ReadSources

class Read:

    origins: Annotated[ReadOrigins,'Classe de leitura de origens']
    sources: Annotated[ReadSources,'Classe de leitura de fontes externas']

    def __init__(self, root: "Root"):
        self.origins = ReadOrigins(root)
        self.sources = ReadSources(root)
