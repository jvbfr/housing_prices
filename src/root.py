from typing import Annotated

from .config import Config

# from .report import Report
# from .write import Write
from .lib import lib
from .make import Make
from .read import Read


class Root:
    """
    Classe central de agregação dos módulos
    """
    config: Annotated[Config,'Módulo de configurações']
    read: Annotated[Read,'Módulo de leituras']
    make: Annotated[Make,'Módulo de construções']

    def __init__(self):
        self.config = Config(self)
        self.read = Read(self)
        self.make = Make(self)
        # self.report = Report(self)
        # self.write = Write(self)
        self.lib = lib  


