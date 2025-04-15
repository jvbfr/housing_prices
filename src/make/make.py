from typing import Annotated
from .make_process import MakeProcess
from .make_transform import MakeTransform

class Make:

    process: Annotated[MakeProcess,'Classe de Processos']
    transform: Annotated[MakeTransform,'Classe de Transformação']

    def __init__(self, root: "Root"):
        self.process = MakeProcess(root)
        self.transform = MakeTransform(root)



