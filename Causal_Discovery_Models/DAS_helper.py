from importlib.metadata import version  # type: ignore
from typing import List, Optional, Tuple, Any, Dict, FrozenSet, Set, Iterable, Protocol, Union, Any, Callable, cast
import networkx as nx

#__version__ = version(__package__)
Column = Union[None, int, float, str, Tuple]
NetworkxGraph = Union[nx.Graph, nx.DiGraph]



