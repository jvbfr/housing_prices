from typing import Annotated

from .config_make import ConfigMake
from .config_read import ConfigRead


class Config:

    read: Annotated[ConfigRead,'Configuração da classe read']
    make: Annotated[ConfigMake,'Configuração da classe make']

    def __init__(self, root = "Root"):
        self.read = ConfigRead(root)
        self.make = ConfigMake(root)
