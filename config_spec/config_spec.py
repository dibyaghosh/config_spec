from functools import partial
import importlib
from typing import Any, Dict, Tuple, TypedDict, Union
import dataclasses


class SpecDict(TypedDict):
    """The output of spec.to_dict() or dict(spec): a JSON-serializable dictionary representation of a function or class with default args and kwargs.

    To instantiate from a SpecDict, use Spec(**spec_dict).instantiate()
    """

    module: str
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    called: bool


@dataclasses.dataclass
class Spec:
    """A JSON-friendly representation of a callable (function or class) with default args and kwargs.

    A spec is first *created*, (potentially serialized in the middle), then *instantiated*.

    Usage:
        >>> from my_module import my_function
        >>> Spec.create(my_function)(arg1=1, arg2=2).instantiate() == my_function(arg1=1, arg2=2)
        >>> Spec.create(my_function).partial(arg1=1, arg2=2).instantiate() == partial(my_function, arg1=1, arg2=2)

        # Specs can be easily serialized
        >>> spec.to_dict()

        # The following are also possible
        >>> Spec.create(my_function).partial(arg1=1, arg2=2) == Spec.create(my_function, arg1=1, arg2=2)
        >>> spec = Spec.create(partial(my_function, arg1=1, arg2=2))
        >>> spec = Spec.create("my_module:my_function", arg1=1, arg2=2)


    module (str): The module the callable is located in
    name (str): The name of the callable in the module
    args (tuple): The args to pass to the callable
    kwargs (dict): The kwargs to pass to the callable
    called (bool): Whether or not the callable should be called upon instantiation.
    """

    module: str
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    called: bool

    def instantiate(self):
        fn = _import_from_string(self.module, self.name)
        fn = partial(fn, *self.args, **self.kwargs)
        if self.called:
            return fn()
        else:
            return fn

    @classmethod
    def create(cls, callable_or_full_name: Union[str, callable], *args, **kwargs) -> "Spec":  # type: ignore
        """Create a module spec from a callable or import string. When this spec is instantiated,
           it will return a partial object with the given args and kwargs.

        Args:
            callable_or_full_name (str or object): Either the object itself or a fully qualified import string
                (e.g. "octo.model.components.transformer:Transformer")
            args (tuple, optional): Passed into callable upon instantiation.
            kwargs (dict, optional): Passed into callable upon instantiation.
        """
        if isinstance(callable_or_full_name, str):
            assert callable_or_full_name.count(":") == 1, (
                "If passing in a string, it must be a fully qualified import string "
                "(e.g. 'octo.model.components.transformer:Transformer')"
            )
            module, name = callable_or_full_name.split(":")
        else:
            if isinstance(callable_or_full_name, partial):
                callable_or_full_name = _compress_partial(callable_or_full_name)
                kwargs = {**callable_or_full_name.keywords, **kwargs}
                args = callable_or_full_name.args + args
                callable_or_full_name = callable_or_full_name.func
            module, name = _infer_full_name(callable_or_full_name)

        return cls(module=module, name=name, args=args, kwargs=kwargs, called=False)

    def partial(self, *args, **kwargs):
        """Add additional args and kwargs to the spec. (Functional, not in-place)"""
        assert (
            not self.called
        ), "Cannot call partial on a spec that has already been called"
        new_args = self.args + args
        new_kwargs = {**self.kwargs, **kwargs}
        return dataclasses.replace(self, args=new_args, kwargs=new_kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Add additional args and kwargs to the spec, and marks it for calling. (Functional, not in-place)

        When this spec is instantiated, it will execute the callable with the given args and kwargs, instead of returning a partial object.
        """
        assert not self.called, "Cannot call a spec that has already been called"
        new_args = self.args + args
        new_kwargs = {**self.kwargs, **kwds}
        return dataclasses.replace(self, args=new_args, kwargs=new_kwargs, called=True)

    def to_dict(self):
        return dataclasses.asdict(self)

    def __iter__(self):
        # so that we can do dict(spec)
        return iter(self.to_dict().items())

    @classmethod
    def from_dense(cls, module, name, args, called, **kwargs):
        """Creates a spec from a dense spec dict, which has kwargs flattened into the top level."""
        return cls(module=module, name=name, args=args, kwargs=kwargs, called=called)


def _infer_full_name(o: object):
    if hasattr(o, "__module__") and hasattr(o, "__qualname__"):
        return o.__module__, o.__qualname__
    else:
        raise ValueError(
            f"Could not infer identifier for {o}. "
            "Please pass in a fully qualified import string instead "
            "e.g. 'octo.model.components.transformer:Transformer'"
        )


def _import_from_string(module_string: str, name: str):
    try:
        module = importlib.import_module(module_string)
        subs = name.split(".")
        o = module
        for sub in subs:
            o = getattr(o, sub)
        return o
    except Exception as e:
        raise ValueError(f"Could not import {module_string}:{name}") from e


def _is_spec_dict(spec_dict: dict):
    return hasattr(spec_dict, "keys") and set(spec_dict.keys()) == set(
        SpecDict.__annotations__.keys()
    )


def _is_dense_spec_dict(spec_dict: dict):
    return hasattr(spec_dict, "keys") and spec_dict.get("__dense_spec__", False)


def _densify_spec_dict(spec_dict: dict):
    """Converts a spec dict to a denser form, by flattening kwargs into ."""
    spec_dict = spec_dict.copy()
    spec_dict.update(spec_dict.pop("kwargs", {}))
    spec_dict["__dense_spec__"] = True
    return spec_dict


def _undensify_spec_dict(spec_dict: dict):
    """Converts a spec dict to a denser form, by flattening kwargs into ."""
    spec_dict = spec_dict.copy()
    assert spec_dict.pop("__dense_spec__", False), "Spec dict is not dense"
    spec_dict["kwargs"] = {
        k: v for k, v in spec_dict.items() if k not in SpecDict.__annotations__
    }
    return spec_dict


def _compress_partial(p: partial):
    """Compresses a nested partial into a single partial object."""
    args = p.args
    kwargs = p.keywords
    fn = p.func
    while isinstance(fn, partial):
        args = fn.args + args
        kwargs = {**fn.keywords, **kwargs}
        fn = fn.func
    return partial(fn, *args, **kwargs)


def asdict(o: object, dense: bool = False):
    """Converting Specs and partial objects to SpecDicts, recursively. Use this when creating a config.

    Args:
        o: Either a partial object, Spec, or a nested (list / dict / tuple) containing partials.
        dense: If true, folds kwargs into spec, making the final dict slightly less deep (default: False)
    Returns:
        An object with the same structure as o, but with all partials replaced with SpecDics (dictionaries).

    """
    if isinstance(o, partial):
        o = _compress_partial(o)  # remove nested partials
        args = asdict(o.args, dense)
        kwargs = asdict(o.keywords, dense)
        fn = o.func
        out = Spec.create(fn, *args, **kwargs).to_dict()
        if dense:
            out = _densify_spec_dict(out)
        return out
    elif isinstance(o, Spec):
        args = asdict(o.args, dense)
        kwargs = asdict(o.kwargs, dense)
        out = dataclasses.replace(o, args=args, kwargs=kwargs).to_dict()
        if dense:
            out = _densify_spec_dict(out)
        return out
    elif hasattr(o, "items"):
        # a little hacky, but so that we can use this on ml_collections.ConfigDicts as well
        return type(o)({k: asdict(v, dense) for k, v in o.items()})
    elif isinstance(o, (list, tuple)):
        return type(o)(asdict(x, dense) for x in o)
    else:
        return o


def recursive_instantiate_spec(o: object):
    """Goes through (a potentially nested) object, turns all spec dictionaries into Specs, and instantiates them.
    Use this when loading from a config dict.

    Args:
        o: Either a Spec object, or a nested (list / dict / tuple) containing SpecDicts.
    Returns:
        An object with the same structure as o, but with all Specs instantiated.
    """
    if _is_dense_spec_dict(o):
        o = _undensify_spec_dict(o)

    if _is_spec_dict(o):
        args = recursive_instantiate_spec(o["args"])
        kwargs = recursive_instantiate_spec(o["kwargs"])
        return Spec(**{**o, "args": args, "kwargs": kwargs}).instantiate()
    elif hasattr(o, "items"):
        # a little hacky, but so that we can use this on ml_collections.ConfigDicts as well
        return type(o)({k: recursive_instantiate_spec(v) for k, v in o.items()})
    elif isinstance(o, (list, tuple)):
        return type(o)(recursive_instantiate_spec(x) for x in o)
    else:
        return o
