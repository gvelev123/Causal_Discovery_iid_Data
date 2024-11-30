from typing import List, Optional, Tuple, Any, Dict, FrozenSet, Set, Iterable, Protocol, Union, Any, Callable, cast
import math
import numpy as np
from numpy.typing import NDArray
from scipy.stats import ttest_ind
import networkx as nx
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

from abc import ABCMeta, abstractmethod
from pygam import LinearGAM, s
from pygam.terms import Term, TermList
from sklearn.metrics.pairwise import rbf_kernel

from dataclasses import dataclass, field
from warnings import warn
import networkx as nx
import inspect
import warnings
from collections import defaultdict
from copy import deepcopy
from copy import copy
from importlib.metadata import version  # type: ignore
import types
from itertools import combinations


CALLABLES = types.FunctionType, types.MethodType

from .DAS_helper import  Column,NetworkxGraph#__version__, 



def kernel_width(X: NDArray):
    """
    Estimate width of the Gaussian kernel.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_nodes)
        Matrix of the data.
    """
    X_diff = np.expand_dims(X, axis=1) - X  # Gram matrix of the data
    D = np.linalg.norm(X_diff, axis=2).flatten()
    D_nonzeros = D[D > 0]  # Remove zeros
    s = np.median(D_nonzeros) if np.any(D_nonzeros) else 1
    return s



def full_dag(top_order: List[int]) -> NDArray:
    """Find adjacency matrix of the fully connected DAG from the topological order.

    Parameters
    ----------
    top_order : List[int]
        Topological order of nodes in a causal graph.

    Returns
    -------
    A : np.ndarray
        The fully connected adjacency matrix encoding the input topological order.
    """
    d = len(top_order)
    A = np.zeros((d, d))
    for i, var in enumerate(top_order):
        A[var, top_order[i + 1 :]] = 1
    return A

# Preliminary Neighbours Search
def pns(A: NDArray, X: NDArray, pns_threshold, pns_num_neighbors) -> NDArray:
    """Preliminary Neighbors Selection (PNS) pruning on adjacency matrix `A`.

    Variable selection preliminary to CAM pruning.
    PNS :footcite:`Buhlmann2013` allows to scale CAM pruning to large graphs
    (~20 or more nodes), with sensitive reduction of computational time.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix representation of a dense graph.
    X : np.ndarray of shape (n_samples, n_nodes)
        Dataset with observations of the causal variables.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float, optional
        Threshold to use for PNS, default is 1.

    Returns
    -------
    A : np.ndarray
        Pruned adjacency matrix.

    References
    ----------
    .. footbibliography::
    """
    num_nodes = X.shape[1]
    for node in range(num_nodes):
        X_copy = np.copy(X)
        X_copy[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(X_copy, X[:, node])
        selected_reg = SelectFromModel(
            reg,
            threshold="{}*mean".format(pns_threshold),
            prefit=True,
            max_features=pns_num_neighbors,
        )
        mask_selected = selected_reg.get_support(indices=False)

        mask_selected = mask_selected.astype(A.dtype)
        A[:, node] *= mask_selected

    return A

def full_adj_to_order(A: NDArray) -> List[int]:
    """Find topological ordering from the adjacency matrix A.

    Parameters
    ----------
    A : np.ndarray
        The fully connected adjacency matrix encoding the input topological order.

    Returns
    -------
    top_order : List[int]
        Topological order encoding of the input fully connected adjacency matrix.
    """
    order = list(A.sum(axis=1).argsort())
    order.reverse()  # reverse to get order starting with source nodes
    return order

class Graph(Protocol):
    """Protocol for graphs to work with dodiscover algorithms."""

    @property
    def nodes(self) -> Iterable:
        """Return an iterable over nodes in graph."""
        pass

    def edges(self, data=None) -> Iterable:
        """Return an iterable over edge tuples in graph."""
        pass

    def has_edge(self, u, v, edge_type="any") -> bool:
        """Check if graph has an edge for a specific edge type."""
        pass

    def add_node(self, node_for_adding, **attr) -> None:
        """Add a node to the graph."""
        pass

    def remove_node(self, u) -> None:
        """Remove a node from the graph."""
        pass

    def remove_edges_from(self, edges) -> None:
        """Remove a set of edges from the graph."""
        pass

    def add_edge(self, u, v, edge_type="all") -> None:
        """Add an edge to the graph."""
        pass

    def remove_edge(self, u, v, edge_type="all") -> None:
        """Remove an edge from the graph."""
        pass

    def neighbors(self, node) -> Iterable:
        """Iterate over all nodes that have any edge connection with 'node'."""
        pass

    def to_undirected(self) -> nx.Graph:
        """Convert a graph to a fully undirected networkx graph.

        All nodes are connected by an undirected edge if there are any
        edges between the two.
        """
        pass

    def subgraph(self, nodes):
        """Get subgraph based on nodes."""
        pass

    def copy(self):
        """Create a copy of the graph."""
        pass

class InconsistentVersionWarning(UserWarning):
    """Warning raised when an estimator is unpickled with a inconsistent version.

    Parameters
    ----------
    estimator_name : str
        Estimator name.

    current_dodiscover_version : str
        Current dodiscover version.

    original_dodiscover_version : str
        Original dodiscover version.
    """

    def __init__(self, *, estimator_name, current_dodiscover_version, original_dodiscover_version):
        self.estimator_name = estimator_name
        self.current_dodiscover_version = current_dodiscover_version
        self.original_dodiscover_version = original_dodiscover_version

    def __str__(self):
        return (
            f"Trying to unpickle estimator {self.estimator_name} from version"
            f" {self.original_dodiscover_version} when "
            f"using version {self.current_dodiscover_version}. This might lead to breaking"
            " code or "
            "invalid results. Use at your own risk. "
        )


class BasePyWhy:
    """Base class for all PyWhy class objects.

    TODO: add parameter validation and data validation from sklearn.
    TODO: add HTML representation.

    Notes
    -----
    All learners and context should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "dodiscover estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this Context.

        TODO: can update this when we build a causal-Pipeline similar to sklearn's Pipeline.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)

            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                # future proof for pipeline objects
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            elif deep and not isinstance(value, type):
                # this ensures a deepcopy is applied, which is useful for graphs
                value = deepcopy(value)

            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : instance
            Learner instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __getstate__(self):
        if getattr(self, "__slots__", None):
            raise TypeError(
                "You cannot use `__slots__` in objects inheriting from "
                "`dodiscover.base.BasePyWhy`."
            )

        try:
            state = super().__getstate__()
            if state is None:
                # For Python 3.11+, empty instance (no `__slots__`,
                # and `__dict__`) will return a state equal to `None`.
                state = self.__dict__.copy()
        except AttributeError:
            # Python < 3.11
            state = self.__dict__.copy()

        if type(self).__module__.startswith("dodiscover."):
            return dict(state.items(), _dodiscover_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        #if type(self).__module__.startswith("dodiscover."):
        #    pickle_version = state.pop("_dodiscover_version", "pre-0.18")
        #    if pickle_version != __version__:
        #        warnings.warn(
        #            InconsistentVersionWarning(
        #                estimator_name=self.__class__.__name__,
        #                current_dodiscover_version=__version__,
        #                original_dodiscover_version=pickle_version,
        #            ),
        #        )
        #try:
        #    super().__setstate__(state)
        #except AttributeError:
        self.__dict__.update(state)

# TODO: we should try to make the thing frozen
# - this would require easy copying of the Context into a new context
# - but resetting e.g. only say one variable like the init_graph
# - IDEAS: perhaps add a function `new_context = copy_context(context, **kwargs)`
# - where kwargs are the things to change.
@dataclass(
    eq=True,
    # frozen=True
)
class Context(BasePyWhy):
    """Context of assumptions, domain knowledge and data.

    This should NOT be instantiated directly. One should instead
    use `dodiscover.make_context` to build a Context data structure.

    Parameters
    ----------
    variables : Set
        Set of observed variables. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    latents : Set
        Set of latent "unobserved" variables. If neither ``latents``,
        nor ``variables`` is set, then it is presumed that ``variables`` consists
        of the columns of ``data`` and ``latents`` is the empty set.
    init_graph : Graph
        The graph to start with.
    included_edges : nx.Graph
        Included edges without direction.
    excluded_edges : nx.Graph
        Excluded edges without direction.
    state_variables : Dict
        Name of intermediate state variables during the learning process.
    intervention_targets : list of tuple
        List of intervention targets (known, or unknown), which correspond to
        the nodes in the graph (known), or indices of datasets that contain
        interventions (unknown).

    Raises
    ------
    ValueError
        ``variables`` and ``latents`` if both set, should contain the set of
        all columns in ``data``.

    Notes
    -----
    Context is a data structure for storing assumptions, domain knowledge,
    priors and other structured contexts alongside the datasets. This class
    is used in conjunction with a discovery algorithm.

    Setting the a priori explicit direction of an edge is not supported yet.

    **Testing for equality**

    Currently, testing for equality is done on all attributes that are not
    graphs. Defining equality among graphs is ill-defined, and as such, we
    leave testing of the internal graphs to users. Some checks of equality
    for example can be :func:`networkx.algorithms.isomorphism.is_isomorphic`
    for checking isomorphism among two graphs.
    """

    observed_variables: Set[Column]
    latent_variables: Set[Column]
    state_variables: Dict[str, Any]
    init_graph: Graph = field(compare=False)
    included_edges: nx.Graph = field(compare=False)
    excluded_edges: nx.Graph = field(compare=False)

    ########################################################
    # for interventional data
    ########################################################
    # the number of distributions we expect to have access to
    num_distributions: int = field(default=1)

    # whether or not observational distribution is present
    obs_distribution: bool = field(default=True)

    # (optional) known intervention targets, corresponding to nodes in the graph
    intervention_targets: List[Tuple[Column]] = field(default_factory=list)

    # (optional) mapping F-nodes to their symmetric difference intervention targets
    symmetric_diff_map: Dict[Any, FrozenSet] = field(default_factory=dict)

    # sigma-map mapping F-nodes to their distribution indices
    sigma_map: Dict[Any, Tuple] = field(default_factory=dict)
    f_nodes: List = field(default_factory=list)

    ########################################################
    # for general multi-domain data
    ########################################################
    # the number of domains we expect to have access to
    num_domains: int = field(default=1)

    # map each augmented node to a tuple of domains (e.g. (0, 1), or (1,))
    domain_map: Dict[Any, Tuple] = field(default_factory=dict)
    s_nodes: List = field(default_factory=list)

    def add_state_variable(self, name: str, var: Any) -> "Context":
        """Add a state variable.

        Called by an algorithm to persist data objects that
        are used in intermediate steps.

        Parameters
        ----------
        name : str
            The name of the state variable.
        var : any
            Any state variable.
        """
        self.state_variables[name] = var
        return self

    def state_variable(self, name: str, on_missing: str = "raise") -> Any:
        """Get a state variable.

        Parameters
        ----------
        name : str
            The name of the state variable.
        on_missing : {'raise', 'warn', 'ignore'}
            Behavior if ``name`` is not in the dictionary of state variables.
            If 'raise' (default) will raise a RuntimeError. If 'warn', will
            raise a UserWarning. If 'ignore', will return `None`.

        Returns
        -------
        state_var : Any
            The state variable.
        """
        if name not in self.state_variables and on_missing != "ignore":
            err_msg = f"{name} is not a state variable: {self.state_variables}"
            if on_missing == "raise":
                raise RuntimeError(err_msg)
            elif on_missing == "warn":
                warn(err_msg)

        return self.state_variables.get(name)

    def copy(self) -> "Context":
        """Create a deepcopy of the context."""
        return Context(**self.get_params(deep=True))

    ###############################################################
    # Methods for interventional data.
    ###############################################################
    def get_non_augmented_nodes(self) -> Set:
        """Get the set of non f-nodes."""
        non_augmented_nodes = set()
        f_nodes = set(self.f_nodes)
        s_nodes = set(self.s_nodes)
        for node in self.init_graph.nodes:
            if node not in f_nodes and node not in s_nodes:
                non_augmented_nodes.add(node)
        return non_augmented_nodes

    def get_augmented_nodes(self) -> Set:
        """Get the set of f-nodes."""
        return set(self.f_nodes).union(set(self.s_nodes))

    def reverse_sigma_map(self) -> Dict:
        """Get the reverse sigma-map."""
        reverse_map = dict()
        for node, mapping in self.sigma_map.items():
            reverse_map[mapping] = node
        return reverse_map




class ContextBuilder:
    """A builder class for creating observational data Context objects ergonomically.

    The ContextBuilder is meant solely to build Context objects that work
    with observational datasets.

    The context builder provides a way to capture assumptions, domain knowledge,
    and data. This should NOT be instantiated directly. One should instead use
    `dodiscover.make_context` to build a Context data structure.
    """

    _init_graph: Optional[Graph] = None
    _included_edges: Optional[NetworkxGraph] = None
    _excluded_edges: Optional[NetworkxGraph] = None
    _observed_variables: Optional[Set[Column]] = None
    _latent_variables: Optional[Set[Column]] = None
    _state_variables: Dict[str, Any] = dict()

    def __init__(self) -> None:
        # perform an error-check on subclass definitions of ContextBuilder
        for attribute, value in self.__class__.__dict__.items():
            if isinstance(value, CALLABLES) or isinstance(value, property):
                continue
            if attribute.startswith("__"):
                continue

            if not hasattr(self, attribute[1:]):
                raise RuntimeError(
                    f"Context objects has class attributes that do not have "
                    f"a matching class method to set the attribute, {attribute}. "
                    f"The form of the attribute must be '_<name>' and a "
                    f"corresponding function name '<name>'."
                )

    def init_graph(self, graph: Graph) -> "ContextBuilder":
        """Set the partial graph to start with.

        Parameters
        ----------
        graph : Graph
            The new graph instance.

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._init_graph = graph
        return self

    def excluded_edges(self, exclude: Optional[NetworkxGraph]) -> "ContextBuilder":
        """Set exclusion edge constraints to apply in discovery.

        Parameters
        ----------
        excluded : Optional[NetworkxGraph]
            Edges that should be excluded in the resultant graph

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        if self._included_edges is not None:
            for u, v in exclude.edges:  # type: ignore
                if self._included_edges.has_edge(u, v):
                    raise RuntimeError(f"{(u, v)} is already specified as an included edge.")
        self._excluded_edges = exclude
        return self

    def included_edges(self, include: Optional[NetworkxGraph]) -> "ContextBuilder":
        """Set inclusion edge constraints to apply in discovery.

        Parameters
        ----------
        included : Optional[NetworkxGraph]
            Edges that should be included in the resultant graph

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        if self._excluded_edges is not None:
            for u, v in include.edges:  # type: ignore
                if self._excluded_edges.has_edge(u, v):
                    raise RuntimeError(f"{(u, v)} is already specified as an excluded edge.")
        self._included_edges = include
        return self

    def edges(
        self,
        include: Optional[NetworkxGraph] = None,
        exclude: Optional[NetworkxGraph] = None,
    ) -> "ContextBuilder":
        """Set edge constraints to apply in discovery.

        Parameters
        ----------
        included : Optional[NetworkxGraph]
            Edges that should be included in the resultant graph
        excluded : Optional[NetworkxGraph]
            Edges that must be excluded in the resultant graph

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._included_edges = include
        self._excluded_edges = exclude
        return self

    def observed_variables(self, observed: Optional[Set[Column]] = None) -> "ContextBuilder":
        """Set observed variables.

        Parameters
        ----------
        observed : Optional[Set[Column]]
            Set of observed variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        """
        if self._latent_variables is not None and any(
            obs_var in self._latent_variables for obs_var in observed  # type: ignore
        ):
            raise RuntimeError(
                f"Latent variables are set already {self._latent_variables}, "
                f'which contain variables you are trying to set as "observed".'
            )
        self._observed_variables = observed
        return self

    def latent_variables(self, latents: Optional[Set[Column]] = None) -> "ContextBuilder":
        """Set latent variables.

        Parameters
        ----------
        latents : Optional[Set[Column]]
            Set of latent "unobserved" variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        """
        if self._observed_variables is not None and any(
            latent_var in self._observed_variables for latent_var in latents  # type: ignore
        ):
            raise RuntimeError(
                f"Observed variables are set already {self._observed_variables}, "
                f'which contain variables you are trying to set as "latent".'
            )
        self._latent_variables = latents
        return self

    def variables(
        self,
        observed: Optional[Set[Column]] = None,
        latents: Optional[Set[Column]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> "ContextBuilder":
        """Set variable-list information to utilize in discovery.

        Parameters
        ----------
        observed : Optional[Set[Column]]
            Set of observed variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        latents : Optional[Set[Column]]
            Set of latent "unobserved" variables, by default None. If neither ``latents``,
            nor ``variables`` is set, then it is presumed that ``variables`` consists
            of the columns of ``data`` and ``latents`` is the empty set.
        data : Optional[pd.DataFrame]
            the data to use for variable inference.

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._observed_variables = observed
        self._latent_variables = latents

        if data is not None:
            (observed, latents) = self._interpolate_variables(data, observed, latents)
            self._observed_variables = observed
            self._latent_variables = latents

        if self._observed_variables is None:
            raise ValueError("Could not infer variables from data or given arguments.")

        return self

    def state_variables(self, state_variables: Dict[str, Any]) -> "ContextBuilder":
        """Set the state variables to use in discovery.

        Parameters
        ----------
        state_variables : Dict[str, Any]
            The state variables to use in discovery.

        Returns
        -------
        self : ContextBuilder
            The builder instance
        """
        self._state_variables = state_variables
        return self

    def state_variable(self, name: str, var: Any) -> "ContextBuilder":
        """Add a state variable.

        Called by an algorithm to persist data objects that
        are used in intermediate steps.

        Parameters
        ----------
        name : str
            The name of the state variable.
        var : any
            Any state variable.
        """
        self._state_variables[name] = var
        return self

    def build(self) -> Context:
        """Build the Context object.

        Returns
        -------
        context : Context
            The populated Context object.
        """
        if self._observed_variables is None:
            raise ValueError("Could not infer variables from data or given arguments.")

        empty_graph = self._empty_graph_func(self._observed_variables)
        return Context(
            init_graph=self._interpolate_graph(self._observed_variables),
            included_edges=self._included_edges or empty_graph(),
            excluded_edges=self._excluded_edges or empty_graph(),
            observed_variables=self._observed_variables,
            latent_variables=self._latent_variables or set(),
            state_variables=self._state_variables,
        )

    def _interpolate_variables(
        self,
        data: pd.DataFrame,
        observed: Optional[Set[Column]] = None,
        latents: Optional[Set[Column]] = None,
    ) -> Tuple[Set[Column], Set[Column]]:
        # initialize and parse the set of variables, latents and others
        columns = set(data.columns)
        if observed is not None and latents is not None:
            if columns - set(observed) != set(latents):
                raise ValueError(
                    "If observed and latents are both set, then they must "
                    "include all columns in data."
                )
        elif observed is None and latents is not None:
            observed = columns - set(latents)
        elif latents is None and observed is not None:
            latents = columns - set(observed)
        elif observed is None and latents is None:
            # when neither observed, nor latents is set, it is assumed
            # that the data is all "not latent"
            observed = columns
            latents = set()

        observed = set(cast(Set[Column], observed))
        latents = set(cast(Set[Column], latents))
        return (observed, latents)

    def _interpolate_graph(self, graph_variables) -> nx.Graph:
        if self._observed_variables is None:
            raise ValueError("Must set variables() before building Context.")

        complete_graph = lambda: nx.complete_graph(graph_variables, create_using=nx.Graph)
        has_all_variables = lambda g: set(g.nodes).issuperset(set(self._observed_variables))

        # initialize the starting graph
        if self._init_graph is None:
            return complete_graph()
        else:
            if not has_all_variables(self._init_graph):
                raise ValueError(
                    f"The nodes within the initial graph, {self._init_graph.nodes}, "
                    f"do not match the nodes in the passed in data, {self._observed_variables}."
                )
            return self._init_graph

    def _empty_graph_func(self, graph_variables) -> Callable:
        empty_graph = lambda: nx.empty_graph(graph_variables, create_using=nx.Graph)
        return empty_graph


class InterventionalContextBuilder(ContextBuilder):
    """A builder class for creating observational+interventional data Context objects.

    The InterventionalContextBuilder is meant solely to build Context objects that work
    with observational + interventional datasets.

    The context builder provides a way to capture assumptions, domain knowledge,
    and data. This should NOT be instantiated directly. One should instead use
    :func:`dodiscover.make_context` to build a Context data structure.

    Notes
    -----
    The number of distributions and/or interventional targets must be set in order
    to build the :class:`~.context.Context` object here.
    """

    _intervention_targets: Optional[List[Tuple[Column]]] = None
    _num_distributions: Optional[int] = None
    _obs_distribution: bool = True

    def obs_distribution(self, has_obs_distrib: bool):
        """Whether or not we have access to the observational distribution.

        By default, this is True and assumed to be the first distribution.
        """
        self._obs_distribution = has_obs_distrib
        return self

    def num_distributions(self, num_distribs: int):
        """Set the number of data distributions we are expected to have access to.

        Note this must include observational too if observational is assumed present.
        To assume that we do not have access to observational data, use the
        :meth:`InterventionalContextBuilder.obs_distribution` to turn off that assumption.

        Parameters
        ----------
        num_distribs : int
            Number of distributions we will have access to. Will set the number of
            distributions to be ``num_distribs + 1`` if ``_obs_distribution is True`` (default).
        """
        self._num_distributions = num_distribs
        return self

    def intervention_targets(self, targets: List[Tuple[Column]]):
        """Set known intervention targets of the data.

        Will also automatically infer the F-nodes that will be present
        in the graph. For more information on F-nodes see ``pywhy-graphs``.

        Parameters
        ----------
        interventions : List of tuple
            A list of tuples of nodes that are known intervention targets.
            Assumes that the order of the interventions marked are those of the
            passed in the data.

            If intervention targets are unknown, then this is not necessary.
        """
        self._intervention_targets = targets
        return self

    def build(self) -> Context:
        """Build the Context object.

        Returns
        -------
        context : Context
            The populated Context object.
        """
        if self._observed_variables is None:
            raise ValueError("Could not infer variables from data or given arguments.")
        if self._num_distributions is None:
            warn(
                "There is no intervention context set. Are you sure you are using "
                "the right contextbuilder? If you only have observational data "
                "use `ContextBuilder` instead of `InterventionContextBuilder`."
            )

        # infer intervention targets and number of distributions
        if self._intervention_targets is None:
            intervention_targets = []
        else:
            intervention_targets = self._intervention_targets
        if self._num_distributions is None:
            num_distributions = int(self._obs_distribution) + len(intervention_targets)
        else:
            num_distributions = self._num_distributions

        # error-check if intervention targets was set that it matches the distributions
        if len(intervention_targets) > 0:
            if len(intervention_targets) + int(self._obs_distribution) != num_distributions:
                raise RuntimeError(
                    f"Setting the number of distributions {num_distributions} does not match the "
                    f"number of intervention targets {len(intervention_targets)}."
                )

        # get F-nodes and sigma-map
        f_nodes, sigma_map, symmetric_diff_map = self._create_augmented_nodes(
            intervention_targets, num_distributions
        )
        graph_variables = set(self._observed_variables).union(set(f_nodes))

        empty_graph = self._empty_graph_func(graph_variables)
        return Context(
            init_graph=self._interpolate_graph(graph_variables),
            included_edges=self._included_edges or empty_graph(),
            excluded_edges=self._excluded_edges or empty_graph(),
            observed_variables=self._observed_variables,
            latent_variables=self._latent_variables or set(),
            state_variables=self._state_variables,
            intervention_targets=intervention_targets,
            f_nodes=f_nodes,
            sigma_map=sigma_map,
            symmetric_diff_map=symmetric_diff_map,
            obs_distribution=self._obs_distribution,
            num_distributions=num_distributions,
        )

    def _create_augmented_nodes(
        self, intervention_targets, num_distributions
    ) -> Tuple[List, Dict, Dict]:
        """Create augmented nodes, sigma map and optionally a symmetric difference map.

        Given a number of distributions attributed to interventions, one constructs
        F-nodes to add to the causal graph via one of two procedures:

        - (known targets): For all pairs of intervention targets, form the
          symmetric difference and then assign this to a new F-node.
          This is ``n_targets choose 2``
        - (unknown targets): For all pairs of incoming distributions, form
          a new F-node. This is ``n_distributions choose 2``

        The difference is the additional information is encoded in the known
        targets case. That is we know the symmetric difference mapping for each
        F-node.

        Returns
        -------
        Tuple[List, Dict[Any, Tuple], Dict[Any, FrozenSet]]
            _description_
        """
        augmented_nodes = []
        sigma_map = dict()
        symmetric_diff_map = dict()

        # add the empty intervention if there is assumed observational data
        if self._obs_distribution:
            distribution_targets_idx = [0]
        else:
            distribution_targets_idx = []

        # now map all distribution targets to their indexed distribution
        int_dist_idx = np.arange(int(self._obs_distribution), num_distributions).tolist()
        distribution_targets_idx.extend(int_dist_idx)

        # store known-targets, which are sets of nodes
        targets = []
        if len(intervention_targets) > 0:
            if self._obs_distribution:
                targets.append(())
            targets.extend(copy(list(intervention_targets)))  # type: ignore

        # create F-nodes, their symmetric difference mapping and sigma-mapping to
        # intervention targets
        for idx, (jdx, kdx) in enumerate(combinations(distribution_targets_idx, 2)):
            f_node = ("F", idx)
            augmented_nodes.append(f_node)
            sigma_map[f_node] = (jdx, kdx)

            # if we additionally know the intervention targets
            if len(intervention_targets) > 0:
                i_target: Set = set(targets[jdx])
                j_target: Set = set(targets[kdx])

                # form symmetric difference and store its frozenset
                # (so that way order is not important)
                f_node_targets = frozenset(i_target.symmetric_difference(j_target))
                symmetric_diff_map[f_node] = f_node_targets
        return augmented_nodes, sigma_map, symmetric_diff_map

    def _interpolate_graph(self, graph_variables) -> nx.Graph:
        init_graph = super()._interpolate_graph(graph_variables)

        # do error-check
        if not all(node in init_graph for node in graph_variables):
            raise RuntimeError(
                "Not all nodes (observational and f-nodes) are part of the init graph."
            )
        return init_graph


def make_context(
    context: Optional[Context] = None, create_using=ContextBuilder
) -> Union[ContextBuilder, InterventionalContextBuilder]:
    """Create a new ContextBuilder instance.

    Returns
    -------
    result : ContextBuilder, InterventionalContextBuilder
        The new ContextBuilder instance

    Examples
    --------
    This creates a context object denoting that there are three observed
    variables, ``(1, 2, 3)``.
    >>> context_builder = make_context()
    >>> context = context_builder.variables([1, 2, 3]).build()

    Notes
    -----
    :class:`~.context.Context` objects are dataclasses that creates a dictionary-like access
    to causal context metadata. Copying relevant information from a Context
    object into a `ContextBuilder` is all supported with the exception of
    state variables. State variables are not copied over. To set state variables
    again, one must build the Context and then call
    :py:meth:`~.context.Context.state_variable`.
    """
    result = create_using()
    if context is not None:
        # we create a copy of the ContextBuilder with the current values
        # in the context
        ctx_params = context.get_params()
        for param, value in ctx_params.items():
            if getattr(result, param, None) is not None:
                getattr(result, param)(value)

    return result






class SteinMixin:
    """
    Implementation of Stein gradient estimator and Stein Hessian estimator and helper methods.
    """

    def hessian(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        """Stein estimator of the Hessian of log p(x).

        The Hessian matrix is efficiently estimated by exploitation of the Stein identity.
        Implements :footcite:`rolland2022`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            I.i.d. samples from p(X) joint distribution.
        eta_G: float
            regularization parameter for ridge regression in Stein gradient estimator.
        eta_H: float
            regularization parameter for ridge regression in Stein hessian estimator.

        Returns
        -------
        H : np.ndarray
            Stein estimator of the Hessian matrix of log p(x).

        References
        ----------
        .. footbibliography::
        """
        _, d = X.shape
        s = kernel_width(X)
        K = self._evaluate_kernel(X, s=s)
        nablaK = self._evaluate_nablaK(K, X, s)
        G = self.score(X, eta_G, K, nablaK)

        # Compute the Hessian by column stacked together
        H = np.stack([self._hessian_col(X, G, col, eta_H, K, s) for col in range(d)], axis=1)
        return H

    def score(
        self,
        X: NDArray,
        eta_G: float,
        K: Optional[NDArray] = None,
        nablaK: Optional[NDArray] = None,
    ) -> NDArray:
        """Stein gradient estimator of the score, i.e. gradient log p(x).

        The Stein gradient estimator :footcite:`Li2017` exploits the Stein identity
        for efficient estimate of the score function.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            I.i.d. samples from p(X) joint distribution.
        eta_G: float
            regularization parameter for ridge regression in Stein gradient estimator.
        K : np.ndarray of shape (n_samples, n_samples)
            Gaussian kernel evaluated at X, by default None. If `K` is None, it is
            computed inside of the method.
        nablaK : np.ndarray of shape (n_samples, )
            <nabla, K> evaluated dot product, by default None. If `nablaK` is None, it is
            computed inside of the method.

        Returns
        -------
        G : np.ndarray
            Stein estimator of the score function.

        References
        ----------
        .. footbibliography::
        """
        n, _ = X.shape
        if K is None:
            s = kernel_width(X)
            K = self._evaluate_kernel(X, s)
            nablaK = self._evaluate_nablaK(K, X, s)

        G = np.matmul(np.linalg.inv(K + eta_G * np.eye(n)), nablaK)
        return G

    def _hessian_col(
        self, X: NDArray, G: NDArray, c: int, eta: float, K: NDArray, s: float
    ) -> NDArray:
        """Stein estimator of a column of the Hessian of log p(x)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data
        G : np.ndarray
            estimator of the score function.
        c : int
            index of the column of interest.
        eta: float
            regularization parameter for ridge regression in Stein hessian estimator
        K : np.ndarray of shape (n_samples, n_samples)
            Gaussian kernel evaluated at X.
        s : float
            Width of the Gaussian kernel.

        Returns
        -------
        H_col : np.ndarray
            Stein estimator of the c-th column of the Hessian of log p(x)
        """
        X_diff = self._X_diff(X)
        n, _, _ = X_diff.shape

        # Stein estimate
        Gv = np.einsum("i,ij->ij", G[:, c], G)
        nabla2vK = np.einsum("ik,ikj,ik->ij", X_diff[:, :, c], X_diff, K) / s**4
        nabla2vK[:, c] -= np.einsum("ik->i", K) / s**2
        H_col = -Gv + np.matmul(np.linalg.inv(K + eta * np.eye(n)), nabla2vK)
        return H_col

    def hessian_diagonal(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        """Stein estimator of the diagonal of the Hessian matrix of log p(x).

        Parameters
        ----------
        X : np.ndarray (n_samples, n_nodes)
            I.i.d. samples from p(X) joint distribution.
        eta_G: float
            regularization parameter for ridge regression in Stein gradient estimator.
        eta_H: float
            regularization parameter for ridge regression in Stein hessian estimator.

        Returns
        -------
        H_diag : np.ndarray
            Stein estimator of the diagonal of the Hessian matrix of log p(x).
        """
        n, _ = X.shape

        # Score function compute
        s = kernel_width(X)
        K = self._evaluate_kernel(X, s)
        nablaK = self._evaluate_nablaK(K, X, s)
        G = self.score(X, eta_G, K, nablaK)

        # Hessian compute
        X_diff = self._X_diff(X)
        nabla2K = np.einsum("kij,ik->kj", -1 / s**2 + X_diff**2 / s**4, K)
        return -(G**2) + np.matmul(np.linalg.inv(K + eta_H * np.eye(n)), nabla2K)

    def _evaluate_kernel(self, X: NDArray, s: float) -> Tuple[NDArray, Union[NDArray, None]]:
        """
        Evaluate Gaussian kernel from data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        s : float
            Width of the Gaussian kernel.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            Evaluated gaussian kernel.
        """
        K = rbf_kernel(X, gamma=1 / (2 * s**2)) / s
        return K

    def _evaluate_nablaK(self, K: NDArray, X: NDArray, s: float):
        """Evaluate <nabla, K> inner product.

        Parameters
        ----------
        K : np.ndarray of shape (n_samples, n_samples)
            Evaluated gaussian kernel of the matrix of the data.
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        s : float
            Width of the Gaussian kernel.

        Returns
        -------
        nablaK : np.ndarray of shape (n_samples, n_nodes)
            Inner product between the Gram matrix of the data and the kernel matrix K.
            To obtain the Gram matrix, for each sample of X, compute its difference
            with all n_samples.
        """
        nablaK = -np.einsum("kij,ik->kj", self._X_diff(X), K) / s**2
        return nablaK

    def _X_diff(self, X: NDArray):
        """For each sample of X, compute its difference with all n samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.

        Returns
        -------
        X_diff : np.ndarray of shape (n_samples, n_samples, n_nodes)
            Matrix of the difference between samples.
        """
        return np.expand_dims(X, axis=1) - X


# -------------------- Class with CAM-pruning implementation --------------------#


class CAMPruning:
    """Class implementing regression based selection of edges of a DAG.

    Implementation of the CAM-pruning method :footcite:`Buhlmann2013`.
    The algorithm performs selection of edges of an input DAG via hypothesis
    testing on the coefficients of a regression model.

    Parameters
    ----------
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.001.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10. Automatically decreased
        in case of insufficient samples.
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, alpha: float = 0.05, n_splines: int = 10, splines_degree: int = 3):
        self.alpha = alpha
        self.n_splines = n_splines
        self.degree = splines_degree

    def prune(
        self, X: NDArray, A_dense: NDArray, G_included: nx.DiGraph, G_excluded: nx.DiGraph
    ) -> NDArray:
        """
        Prune the dense adjacency matrix `A_dense` from spurious edges.

        Use sparse regression over the matrix of the data `X` for variable selection over the
        edges in the dense (potentially fully connected) adjacency matrix `A_dense`

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        A_dense : np.ndarray of shape (n_nodes, n_nodes)
            Dense adjacency matrix to be pruned.
        G_included : nx.DiGraph
            Graph with edges that are required to be included in the output.
            It encodes assumptions and prior knowledge about the causal graph.
        G_excluded : nx.DiGraph
            Graph with edges that are required to be excluded from the output.
            It encodes assumptions and prior knowledge about the causal graph.

        Returns
        -------
        A : np.ndarray
            The pruned adjacency matrix output of the causal discovery algorithm.
        """
        _, d = X.shape
        A = np.zeros((d, d))
        order = full_adj_to_order(A_dense)
        for c in range(d):
            pot_parents = []
            for p in order[: order.index(c)]:
                if ((not G_excluded.has_edge(p, c)) and A_dense[p, c] == 1) or G_included.has_edge(
                    p, c
                ):
                    pot_parents.append(p)
            if len(pot_parents) > 0:
                parents = self._variable_selection(
                    X[:, pot_parents], X[:, c], pot_parents, c, G_included
                )
                for parent in parents:
                    A[parent, c] = 1

        return A

    def _variable_selection(
        self,
        X: NDArray,
        y: NDArray,
        pot_parents: List[int],
        child: int,
        G_included: nx.DiGraph,
    ) -> List[int]:
        """
        Regression for parents variables selection.

        Implementation of parents selection for `child` node.
        Returns parents of node `child` associated to sample `y`.

        Parameters
        ----------
        X : np.ndarray
            Exogenous variables.
        y : np.ndarray
            Endogenous variables.
        pot_parents: List[int]
            List of potential parents admitted by the topological ordering.
        child : int
            Child node with `pot_parents` potential parent nodes.
        G_included : nx.DiGraph
            Graph with edges that are required to be included in the output.
            It encodes assumptions and prior knowledge about the causal graph.

        Returns
        -------
        parents : List[int]
            The list of selected parents for the input child node.
        """
        _, d = X.shape

        gam = self._fit_gam_model(X, y)
        pvalues = gam.statistics_["p_values"]

        parents = []
        for j in range(d):
            if pvalues[j] < self.alpha or G_included.has_edge(pot_parents[j], child):
                parents.append(pot_parents[j])
        return parents

    def _fit_gam_model(self, X: NDArray, y: NDArray) -> LinearGAM:
        """
        Fit GAM on `X` and `y`.

        Parameters
        ----------
        X : np.ndarray
            exogenous variables.
        y : np.ndarray
            endogenous variables.

        Returns
        -------
        gam : LinearGAM
            Fitted GAM with tuned lambda.

        Notes
        -----
        See https://pygam.readthedocs.io/en/latest/api/lineargam.html
        for implementation details of `LinearGAM`.
        """
        lambda_grid = {1: [0.1, 0.5, 1], 20: [5, 10, 20], 100: [50, 80, 100]}

        try:
            n, d = X.shape
        except ValueError:
            raise ValueError(
                (
                    f"not enough values to unpack (expected 2, got {len(X.shape)}). "
                    + "If vector X has only 1 dimension, try X.reshape(-1, 1)"
                )
            )

        n_splines = self._compute_n_splines(n, d)

        # Preliminary search of the best lambda between lambda_grid.keys()
        splines = [s(i, n_splines=n_splines, spline_order=self.degree) for i in range(d)]
        formula = self._make_formula(splines)
        lam_keys = lambda_grid.keys()
        gam = LinearGAM(formula, fit_intercept=False).gridsearch(
            X, y, lam=lam_keys, progress=False, objective="GCV"
        )
        lambdas = np.squeeze([s.get_params()["lam"] for s in gam.terms], axis=1)

        # Search the best lambdas according to the preliminary search, and get the fitted model
        lam = np.squeeze([lambda_grid[value] for value in lambdas])
        gam = LinearGAM(formula, fit_intercept=False).gridsearch(
            X, y, lam=lam.transpose(), progress=False, objective="GCV"
        )
        return gam

    def _compute_n_splines(self, n: int, d: int) -> int:
        """
        Compute the number of splines used for GAM fitting.

        During GAM fitting, decrease number of splines in case of small sample size.

        Parameters
        ----------
        n : int
            Number of samples in the datasets.
        d : int
            Number of nodes in the graph.

        Returns
        -------
        n_splines : int
            Updated number of splines for GAM fitting.
        """
        n_splines = self.n_splines
        if n / d < 3 * self.n_splines:
            n_splines = math.ceil(n / (3 * self.n_splines))
            print(
                (
                    f"Changed number of basis functions to {n_splines} in order to have"
                    + " enough samples per basis function"
                )
            )
            if n_splines <= self.degree:
                warnings.warn(
                    (
                        f"n_splines must be > spline_order. found: n_splines = {n_splines}"
                        + f" and spline_order = {self.degree}."
                        + f" n_splines set to {self.degree + 1}"
                    )
                )
                n_splines = self.degree + 1
        return n_splines

    def _make_formula(self, splines_list: List[Term]) -> TermList:
        """
        Make formula for PyGAM model from list of splines Term objects.

        The method defines a linear combination of the spline terms.

        Parameters
        ----------
        splines_list : List[Term]
            List of splines terms for the GAM formula.
            Example: [s(0), s(1), s(2)] where s is a B-spline Term from pyGAM.
            The equivalent R formula would be "s(0) + s(1) + s(2)", while the y target
            is provided directly at gam.learn_graph() call.

        Returns
        -------
        terms : TermList
            Formula of the type requested by pyGAM Generalized Additive Models class.

        Notes
        -----
        See https://pygam.readthedocs.io/en/latest/dev-api/terms.html
        for details on `Term` and `TermsList` implementations.
        """
        terms = TermList()
        for spline in splines_list:
            terms += spline
        return terms


# -------------------- Topological ordering interface -------------------- #


class TopOrderInterface(metaclass=ABCMeta):
    """
    Interface for causal discovery based on estimate of topologial ordering and DAG pruning.
    """

    @abstractmethod
    def learn_graph(self, data: pd.DataFrame, context: Optional[Context] = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _top_order(self, X: NDArray) -> Tuple[NDArray, List[int]]:
        raise NotImplementedError()

    @abstractmethod
    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        raise NotImplementedError()


# -------------------- Base class for order-based inference --------------------#
class BaseTopOrder(CAMPruning, TopOrderInterface):
    """Base class for order-based causal discovery.

    Implementation of `TopOrderInterface` defining `fit` method for causal discovery.
    Class inheriting from `BaseTopOrder` need to implement the `top_order` method for inference
    of the topological ordering of the nodes in the causal graph.
    The resulting fully connected matrix is pruned by `prune` method implementation of
    CAM pruning :footcite:`Buhlmann2013`.

    Parameters
    ----------
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.001.
    prune : bool, optional
        If True (default), apply CAM-pruning after finding the topological order.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10. Automatically decreased
        in case of insufficient samples.
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.
    pns : bool, optional
        If True, perform Preliminary Neighbour Search (PNS). Default is False.
        Allows scaling CAM pruning and ordering to large graphs.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float, optional
        Threshold to use for PNS, default is 1.

    Attributes
    ----------
    graph_ : nx.DiGraph
        Adjacency matrix representation of the inferred causal graph.
    order_ : List[int]
        Topological order of the nodes from source to leaf.
    order_graph_ : nx.DiGraph
        Fully connected adjacency matrix representation of the
        inferred topological order.
    labels_to_nodes : Dict[Union[str, int], int]
        Map from the custom node's label  to the node's label by number.
    nodes_to_labels : Dict[int, Union[str, int]]
        Map from the node's label by number to the custom node's label.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        alpha: float = 0.05,
        prune: bool = True,
        n_splines: int = 10,
        splines_degree: int = 3,
        pns: bool = False,
        pns_num_neighbors: Optional[int] = None,
        pns_threshold: float = 1,
    ):
        # Initialize CAMPruning
        super().__init__(alpha, n_splines, splines_degree)

        # Parameters
        self.apply_pruning = prune
        self.do_pns = pns
        self.pns_num_neighbors = pns_num_neighbors
        self.pns_threshold = pns_threshold

        # Attributes
        self.graph_: nx.DiGraph = nx.empty_graph()
        self.order_: List[int] = list()
        self.order_graph_: nx.DiGraph = nx.empty_graph()
        self.labels_to_nodes: Dict[Union[str, int], int] = dict()
        self.nodes_to_labels: Dict[int, Union[str, int]] = dict()

    def _get_leaf(self, leaf: int, remaining_nodes: List[int], current_order: List[int]) -> int:
        """Get leaf node from the list of `remaining_nodes` without an assigned order.

        Parameters
        ----------
        leaf : int
            Leaf position in the list of `remaining_nodes`.
        remaining_nodes : List[int]
            List of nodes without a position in the order.
        current_order : List[int]
            Partial topological order.

        Returns
        -------
        leaf : int
            Leaf index in the list of graph nodes.
        """
        # descendants enforced by edges in self.context and not in the order are used as leaf
        leaf_descendants = self.order_constraints[remaining_nodes[leaf]]
        if not set(leaf_descendants).issubset(set(current_order)):
            k = 0
            while True:
                if leaf_descendants[k] not in current_order:
                    leaf = remaining_nodes.index(leaf_descendants[k])
                    break  # exit when leaf is found
                k += 1
        return leaf

    def learn_graph(self, data_df: pd.DataFrame, context: Optional[Context] = None) -> None:
        """
        Fit topological order based causal discovery algorithm on input data.

        Parameters
        ----------
        data_df : pd.DataFrame
            Datafame of the input data.
        context: Context
            The context of the causal discovery problem.
        """
        if context is None:
            # make a private Context object to store causal context used in this algorithm
            # store the context
            #from dodiscover.context_builder import make_context

            context = make_context().build()

        X = data_df.to_numpy()
        self.context = context

        # Data structure to exchange labels with nodes number
        self.nodes_to_labels = {i: data_df.columns[i] for i in range(len(data_df.columns))}
        self.labels_to_nodes = {data_df.columns[i]: i for i in range(len(data_df.columns))}

        # Check acyclicity condition on included_edges
        self._dag_check_included_edges()
        self.order_constraints = self._included_edges_order_constraints()

        # Inference of the causal order.
        A_dense, order = self._top_order(X)
        self.order_ = order
        order_graph_ = nx.from_numpy_array(full_dag(order), create_using=nx.DiGraph)

        # Inference of the causal graph via pruning
        if self.apply_pruning:
            A = self._prune(X, A_dense)
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        else:
            G = nx.from_numpy_array(full_dag(order), create_using=nx.DiGraph)  # order_graph

        # Relabel the nodes according to the input data_df columns
        self.graph_ = self._postprocess_output(G)
        self.order_graph_ = self._postprocess_output(order_graph_)

    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """
        Prune the dense adjacency matrix `A_dense` from spurious edges.

        Use sparse regression over the matrix of the data `X` for variable selection over the
        edges in the dense (potentially fully connected) adjacency matrix `A_dense`

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        A_dense : np.ndarray of shape (n_nodes, n_nodes)
            Dense adjacency matrix to be pruned.
        Returns
        -------
        A : np.ndarray
            The pruned adjacency matrix output of the causal discovery algorithm.
        """
        G_included = self._get_included_edges_graph()
        G_excluded = self._get_excluded_edges_graph()
        return super().prune(X, A_dense, G_included, G_excluded)

    def _dag_check_included_edges(self) -> None:
        """Check that the edges included in `self.context` does not violate DAG condition."""
        is_dag = nx.is_directed_acyclic_graph(self._get_included_edges_graph())
        if nx.is_empty(self._get_included_edges_graph()):
            is_dag = True
        if not is_dag:
            raise ValueError("Edges included in the graph violate the acyclicity condition!")

    def _included_edges_order_constraints(self) -> Dict[int, List[int]]:
        """For each node find the predecessors enforced by the edges included in `self.context`.

        Returns
        -------
        descendants : Dict[int, List[int]]
            Dictionary with index of a node of the graph as key, list of the descendants of the
            node enforced by self.context.included_edges as value.
        """
        adj = nx.to_numpy_array(self._get_included_edges_graph())
        d, _ = adj.shape
        descendants: Dict = {i: list() for i in range(d)}
        for row in range(d):
            for col in range(d):
                if adj[row, col] == 1:
                    row_descendants = descendants[row]
                    row_descendants.append(col)
                    descendants[row] = row_descendants
        return descendants

    def _postprocess_output(self, graph):
        """Relabel the graph nodes with the custom labels of the input dataframe.

        Parameters
        ----------
        graph : nx.DiGraph
            Networkx directed graph with nodes to relabel.

        Returns
        -------
        G : nx.DiGraph
            Graph with the relabeled nodes.
        """
        G = nx.relabel_nodes(graph, mapping=self.nodes_to_labels)
        return G

    def _get_included_edges_graph(self):
        """Get the `self.context.included_edges` graph with numerical label of the nodes.

        The returned directed graph of the included edges, with one node for each column
        in the dataframe input of the `fit` method, and numerical label of the nodes.

        Returns
        -------
        G : nx.DiGraph
            networkx directed graph of the edges included in `self.context`.
        """
        num_nodes = len(self.labels_to_nodes)
        G = nx.empty_graph(n=num_nodes, create_using=nx.DiGraph)
        for edge in self.context.included_edges.edges():
            u, v = self.labels_to_nodes[edge[0]], self.labels_to_nodes[edge[1]]
            G.add_edge(u, v)
        return G

    def _get_excluded_edges_graph(self):
        """Get the `self.context.excluded_edges` graph with numerical label of the nodes.

        The returned directed graph of the excluded edges, with one node for each column
        in the dataframe input of the `fit` method, and numerical label of the nodes.

        Returns
        -------
        G : nx.DiGraph
            networkx directed graph of the edges excluded in `self.context`.
        """
        num_nodes = len(self.labels_to_nodes)
        G = nx.empty_graph(n=num_nodes, create_using=nx.DiGraph)
        for edge in self.context.excluded_edges.edges():
            u, v = self.labels_to_nodes[edge[0]], self.labels_to_nodes[edge[1]]
            G.add_edge(u, v)
        return G





class SCORE(BaseTopOrder, SteinMixin):
    """The SCORE algorithm for causal discovery.

    SCORE :footcite:`rolland2022` iteratively defines a topological ordering finding leaf
    nodes by comparison of the variance terms of the diagonal entries of the Hessian of
    the log likelihood matrix. Then it prunes the fully connected DAG representation of
    the ordering with CAM pruning :footcite:`Buhlmann2013`.
    The method assumes Additive Noise Model and Gaussianity of the noise terms.

    Parameters
    ----------
    eta_G: float, optional
        Regularization parameter for Stein gradient estimator, default is 0.001.
    eta_H : float, optional
        Regularization parameter for Stein Hessian estimator, default is 0.001.
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.05.
    prune : bool, optional
        If True (default), apply CAM-pruning after finding the topological order.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10.
        Automatically decreased in case of insufficient samples
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.
    estimate_variance : bool, optional
        If True, store estimates the variance of the noise terms from the diagonal of
        Stein Hessian estimator. Default is False.
    pns : bool, optional
        If True, perform Preliminary Neighbour Search (PNS) before CAM pruning step,
        default is False. Allows scaling CAM pruning and ordering to large graphs.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS. If None (default) use all variables.
    pns_threshold: float, optional
        Threshold to use for PNS, default is 1.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Prior knowledge about the included and excluded directed edges in the output DAG
    is supported. It is not possible to provide explicit constraints on the relative
    positions of nodes in the topological ordering. However, explicitly including a
    directed edge in the DAG defines an implicit constraint on the relative position
    of the nodes in the topological ordering (i.e. if directed edge `(i,j)` is
    encoded in the graph, node `i` will precede node `j` in the output order).
    """

    def __init__(
        self,
        eta_G: float = 0.001,
        eta_H: float = 0.001,
        alpha: float = 0.05,
        prune: bool = True,
        n_splines: int = 10,
        splines_degree: int = 3,
        estimate_variance=False,
        pns: bool = False,
        pns_num_neighbors: Optional[int] = None,
        pns_threshold: float = 1,
    ):
        super().__init__(
            alpha, prune, n_splines, splines_degree, pns, pns_num_neighbors, pns_threshold
        )
        self.eta_G = eta_G
        self.eta_H = eta_H
        self.var: List[float] = list()  # data structure for estimated variance of SCM noise terms
        self.estimate_variance = estimate_variance

    def _top_order(self, X: NDArray) -> Tuple[NDArray, List[int]]:
        """Find the topological ordering of the causal variables from X dataset.

        Parameters
        ----------
        X : np.ndarray
            Dataset of observations of the causal variables

        Returns
        -------
        A_dense : np.ndarray
            Fully connected matrix admitted by the topological ordering
        order : List[int]
            Inferred causal order
        """

        def _estimate_var(stein_instance, data):
            H_diag = stein_instance.hessian_diagonal(data, self.eta_G, self.eta_H)
            H_diag = H_diag / H_diag.mean(axis=0)
            return 1 / H_diag[:, 0].var(axis=0).item()

        _, d = X.shape
        order: List[int] = list()
        active_nodes = list(range(d))
        stein = SteinMixin()
        for _ in range(d - 1):
            X = self._find_leaf_iteration(stein, X, active_nodes, order)
        order.append(active_nodes[0])
        if self.estimate_variance:
            self.var.append(_estimate_var(stein, X))
        order.reverse()
        return full_dag(order), order

    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """Pruning of the fully connected adj. matrix representation of the inferred order.

        If self.do_pns = True or self.do_pns is None and number of nodes >= 20, then
        Preliminary Neighbors Search is applied before CAM pruning.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_nodes)
            Matrix of the data.
        A_dense : np.ndarray of shape (n_nodes, n_nodes)
            Dense adjacency matrix to be pruned.

        Returns
        -------
        A : np.ndarray
            The pruned adjacency matrix output of the causal discovery algorithm.
        """
        d = A_dense.shape[0]
        if (self.do_pns) or (self.do_pns is None and d > 20):
            A_dense = pns(
                A=A_dense,
                X=X,
                pns_threshold=self.pns_threshold,
                pns_num_neighbors=self.pns_num_neighbors,
            )
        return super()._prune(X, A_dense)

    def _find_leaf_iteration(
        self, stein_instance: SteinMixin, X: NDArray, active_nodes: List[int], order: List[int]
    ) -> NDArray:
        """Find a leaf by inspection of the diagonal elements of the Hessian of the log-likelihood.

        Leaf node equals the 'argmin' of the variance of the diagonal terms of the Hessian of
        the log-likelihood.
        After a leaf is identified, it is added to the topological order, and the list of nodes
        without position in the ordering is updated.

        Parameters
        ----------
        stein_instance : SteinMixin
            Instance of the Stein estimator with helper methods.
        X : np.ndarray
            Matrix of the data (active nodes only).
        active_nodes : List[int]
            List of the nodes without a position in the topological ordering.
        order : List[int]
            List of partial order.

        Returns
        -------
        X : np.ndarray
            Matrix of the data without the column corresponding to the identified leaf node.
        """
        H_diag = stein_instance.hessian_diagonal(X, self.eta_G, self.eta_H)
        leaf = int(H_diag.var(axis=0).argmin())
        leaf = self._get_leaf(
            leaf, active_nodes, order
        )  # get leaf consistent with order imposed by included_edges from self.context
        order.append(active_nodes[leaf])
        active_nodes.pop(leaf)
        X = np.hstack([X[:, 0:leaf], X[:, leaf + 1 :]])
        if self.estimate_variance:
            self.var.append(1 / H_diag[:, leaf].var(axis=0).item())
        return X

class DAS(SCORE):
    """The DAS (Discovery At Scale) algorithm for causal discovery.

    DAS :footcite:`Montagna2023a` infer the topological ordering using
    SCORE :footcite:`rolland2022`.
    Then it finds edges in the graph by inspection of the non diagonal entries of the
    Hessian of the log likelihood.
    A final, computationally cheap, pruning step is performed with
    CAM pruning :footcite:`Buhlmann2013`.
    The method assumes Additive Noise Model and Gaussianity of the noise terms.
    DAS is a highly scalable method, allowing to run causal discovery on thousands of nodes.
    It reduces the computational complexity of the pruning method of an order of magnitude with
    respect to SCORE.

    Parameters
    ----------
    eta_G: float, optional
        Regularization parameter for Stein gradient estimator, default is 0.001.
    eta_H : float, optional
        Regularization parameter for Stein Hessian estimator, default is 0.001.
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.05.
    prune : bool, optional
        If True (default), apply CAM-pruning after finding the topological order.
        If False, DAS is equivalent to SCORE.
    das_cutoff : float, optional
        Alpha value for hypothesis testing in preliminary DAS pruning.
        If None (default), it is set equal to `alpha`.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10.
        Automatically decreased in case of insufficient samples
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.
    min_parents : int, optional
        Minimum number of edges retained by DAS preliminary pruning step, default is 5.
        min_parents <= 5 doesn't significantly affects execution time, while increasing the
        accuracy.
    max_parents : int, optional
        Maximum number of parents allowed for a single node, default is 20.
        Given that CAM pruning is inefficient for > ~20 nodes, larger values are not advised.
        The value of max_parents should be decrease under the assumption of sparse graphs.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Prior knowledge about the included and excluded directed edges in the output DAG
    is supported. It is not possible to provide explicit constraints on the relative
    positions of nodes in the topological ordering. However, explicitly including a
    directed edge in the DAG defines an implicit constraint on the relative position
    of the nodes in the topological ordering (i.e. if directed edge `(i,j)` is
    encoded in the graph, node `i` will precede node `j` in the output order).
    """

    def __init__(
        self,
        eta_G: float = 0.001,
        eta_H: float = 0.001,
        alpha: float = 0.05,
        prune: bool = True,
        das_cutoff: Optional[float] = None,
        n_splines: int = 10,
        splines_degree: int = 3,
        min_parents: int = 5,
        max_parents: int = 20,
    ):
        super().__init__(
            eta_G, eta_H, alpha, prune, n_splines, splines_degree, estimate_variance=True, pns=False
        )
        self.min_parents = min_parents
        self.max_parents = max_parents
        self.das_cutoff = alpha if das_cutoff is None else das_cutoff

    def _prune(self, X: NDArray, A_dense: NDArray) -> NDArray:
        """
        DAS preliminary pruning of A_dense matrix representation of a fully connected graph.

        Parameters
        ----------
        X : np.ndarray
            n x d matrix of the data
        A_dense : np.ndarray
            fully connected matrix corresponding to a topological ordering

        Returns
        -------
        np.ndarray
            Sparse adjacency matrix representing the pruned DAG.
        """
        _, d = X.shape
        order = full_adj_to_order(A_dense)
        max_parents = self.max_parents + 1  # +1 to account for A[l, l] = 1
        remaining_nodes = list(range(d))
        A_das = np.zeros((d, d))

        hess = self.hessian(X, eta_G=self.eta_G, eta_H=self.eta_H)
        for i in range(d - 1):
            leaf = order[::-1][i]
            hess_l = hess[:, leaf, :][:, remaining_nodes]
            hess_m = np.abs(np.median(hess_l * self.var[leaf], axis=0))
            max_parents = min(max_parents, len(remaining_nodes))

            # Find index of the reference for the hypothesis testing
            topk_indices = np.argsort(hess_m)[::-1][:max_parents]
            topk_values = hess_m[topk_indices]  # largest
            argmin = topk_indices[np.argmin(topk_values)]

            # Edges selection step
            parents = []
            hess_l = np.abs(hess_l)
            l_index = remaining_nodes.index(
                leaf
            )  # leaf index in the remaining nodes (from 0 to len(remaining_nodes)-1)
            for j in range(max_parents):
                node = topk_indices[j]
                if node != l_index:  # enforce diagonal elements = 0
                    if j < self.min_parents:  # do not filter minimum number of parents
                        parents.append(remaining_nodes[node])
                    else:  # filter potential parents with hp testing
                        # Use hess_l[:, argmin] as sample from a zero mean population
                        # (implicit assumption: argmin corresponds to zero mean hessian entry)
                        _, pval = ttest_ind(
                            hess_l[:, node],
                            hess_l[:, argmin],
                            alternative="greater",
                            equal_var=False,
                        )
                        if pval < self.das_cutoff:
                            parents.append(remaining_nodes[node])

            A_das[parents, leaf] = 1
            remaining_nodes.pop(l_index)

        return super()._prune(X, A_das)




