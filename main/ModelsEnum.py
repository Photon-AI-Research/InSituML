from enum import Enum, auto


class ModelsEnum(Enum):
    MLP = auto()
    Autoencoder2D = auto()
    Autoencoder3D = auto()


class TaskEnum(Enum):
    OTHER = auto()
    E_FIELD = auto()
