"""This module provides a Spec class, which is a JSON-friendly representation of a callable (function or class) with default args and kwargs.

When converted to a dictionary (e.g. via dict(spec)), it will look like this:
>>> spec = Spec(torch.optim.Adam, lr=1e-3)
<Spec: functools.partial(torch.optim.adam:Adam, lr=0.0001)>
>>> d = dict(spec)
{
    'target': "torch.optim.adam:Adam" # A fully qualified import name
    'lr': 1e-3,
    '__config_spec__': True # Indicates that this is a SpecDict, and not a regular dict
}
>>> spec.instantiate() ==  Spec.instantiate_from_dict(d) == functools.partial(torch.optim.Adam, lr=1e-3)
"""
from functools import partial
import importlib
from typing import Any, Dict, Tuple, TypedDict, Union, Callable
import dataclasses


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


    target (str): The fully-qualified name of a callable (e.g. "torch.optim:Adam")
    args (tuple): The args to pass to the callable
    kwargs (dict): The kwargs to pass to the callable
    called (bool): Whether to return a functools.partial object (called=False) or to call it (called=True)
    """

    target: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    called: bool = False

    def instantiate(self, recursive: bool = True):
        """Instantiate the spec, by importing the module and partial/calling the callable with the given args and kwargs.

        If recursive=True, will recursively instantiate any nested specs as well in the args and kwargs.
        """
        if recursive:
            return _recursive_instantiate_spec(self)

        fn = _import_from_string(self.target)
        fn = partial(fn, *self.args, **self.kwargs)
        if self.called:
            return fn()
        else:
            return fn

    def __init__(self, target: Union[str, callable], *args, **kwargs):
        """Create a spec from a callable. Note: you can also do the standard dataclass init

        Args:
            target (object): A Callable
            args (tuple, optional): Passed into callable upon instantiation.
            kwargs (dict, optional): Passed into callable upon instantiation.
        """

        if isinstance(target, str):
            assert (
                target.count(":") == 1
            ), "target must be a fully qualified import string (e.g. 'torch.optim:Adam')"
            # Do the dataclass standard init
            self._standard_init(target, *args, **kwargs)
        else:
            self._init_from_callable(target, *args, **kwargs)

    @classmethod
    def from_dict(cls, o: object):
        """Recurses through a nested (list / dict / tuple), converting all SpecDicts to Specs."""
        return _recursive_from_dict(o)

    @classmethod
    def instantiate_from_dict(o: object):
        """Recurses through a nested (list / dict / tuple), converting all SpecDicts to Specs, then instantiates them."""
        return _recursive_instantiate_spec(_recursive_from_dict(o))

    def _standard_init(self, *args, **kwargs):
        if kwargs.get("__config_spec__", False):
            kwargs = _undensify_spec_dict(kwargs)

        for n, field in enumerate(dataclasses.fields(self)):
            value = args[n] if n < len(args) else kwargs.get(field.name, field.default)
            if value == dataclasses.MISSING:
                if field.default_factory != dataclasses.MISSING:
                    value = field.default_factory()
                else:
                    raise Exception(f"Missing required argument {field.name} for Spec")
            setattr(self, field.name, value)

    def _init_from_callable(self, target: Callable, *args, **kwargs) -> "Spec":  # type: ignore
        """Create a module spec from a callable or import string. When this spec is instantiated,
           it will return a partial object with the given args and kwargs.

        Args:
            callable_or_full_name (str or object): Either the object itself or a fully qualified import string
                (e.g. "octo.model.components.transformer:Transformer")
            args (tuple, optional): Passed into callable upon instantiation.
            kwargs (dict, optional): Passed into callable upon instantiation.
        """
        if isinstance(target, partial):
            target = _compress_partial(target)
            kwargs = {**target.keywords, **kwargs}
            args = target.args + args
            target = target.func
        module, name = _infer_full_name(target)
        target_name = f"{module}:{name}"

        return self._standard_init(target_name, args=args, kwargs=kwargs)

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

    def asdict(self, recursive=True, dense=True):
        if recursive:
            return asdict(self, dense=dense)
        else:
            out = dataclasses.asdict(self)
            if dense:
                out = _densify_spec_dict(out)
            return out

    def __iter__(self):
        # so that we can do dict(spec)
        return iter(self.asdict().items())

    def __repr__(self) -> str:
        args = [repr(x) for x in self.args]
        args.extend(f"{k}={v!r}" for (k, v) in self.kwargs.items())
        if not self.called:
            return f"<Spec: functools.partial({self.target}, {', '.join(args)})>"
        else:
            return f"<Spec: {self.target}({', '.join(args)})>"


_spec_field_names = [field.name for field in dataclasses.fields(Spec)]


def _infer_full_name(o: object):
    if hasattr(o, "__module__") and hasattr(o, "__qualname__"):
        return o.__module__, o.__qualname__
    else:
        raise ValueError(
            f"Could not infer identifier for {o}. "
            "Please pass in a fully qualified import string instead "
            "e.g. 'octo.model.components.transformer:Transformer'"
        )


def _import_from_string(target: str):
    """
    Args:
        target: A fully qualified import string (e.g. "torch.optim:Adam")
    """
    module_string, name = target.split(":")
    try:
        module = importlib.import_module(module_string)
        subs = name.split(".")
        o = module
        for sub in subs:
            o = getattr(o, sub)
        return o
    except Exception as e:
        raise ValueError(f"Could not import {target}") from e


def _is_spec_dict(spec_dict: dict):
    return hasattr(spec_dict, "keys") and set(spec_dict.keys()) == set(
        [f.name for f in dataclasses.fields(Spec)]
    )


def _is_dense_spec_dict(spec_dict: dict):
    return hasattr(spec_dict, "keys") and spec_dict.get("__config_spec__", False)


def _densify_spec_dict(spec_dict: dict):
    """Converts a spec dict to a denser form, by
    1) flattening kwargs into the dict
    2) removing default options (e.g. args=(), kwargs={}, called=False)

    """

    spec_dict = spec_dict.copy()
    kwargs = spec_dict.pop("kwargs", {})
    flattenable_kwargs = {k: v for k, v in kwargs.items() if k not in _spec_field_names}
    unflattenable_kwargs = {
        k: v for k, v in kwargs.items() if k in _spec_field_names
    }  # These would name conflict with attributes of our Spec

    spec_dict.update(flattenable_kwargs)
    spec_dict["kwargs"] = unflattenable_kwargs

    # Remove default options (will add them back when we undensify)
    if len(spec_dict["args"]) == 0:
        spec_dict.pop("args")
    if len(spec_dict["kwargs"]) == 0:
        spec_dict.pop("kwargs")
    if spec_dict["called"] is False:
        spec_dict.pop("called")

    spec_dict["__config_spec__"] = True
    return spec_dict


def _undensify_spec_dict(spec_dict: dict):
    """Reverts a densified spec dict back to its original form"""
    spec_dict = spec_dict.copy()
    assert spec_dict.pop("__config_spec__", False), "Spec dict is not dense"

    if "args" not in spec_dict:
        spec_dict["args"] = ()

    if "kwargs" not in spec_dict:
        spec_dict["kwargs"] = {}

    if "called" not in spec_dict:
        spec_dict["called"] = False

    for k in spec_dict.keys():
        if k not in _spec_field_names:
            spec_dict["kwargs"][k] = spec_dict.pop(k)

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


def asdict(o: object, dense: bool = True):
    """Converting Specs and partial objects to SpecDicts, recursively. Use this when creating a config.

    Args:
        o: Either a partial object, Spec, or a nested (list / dict / tuple) containing partials.
        dense: If true, folds kwargs into spec, making the final dict slightly less deep (default: False)
    Returns:
        An object with the same structure as o, but with all partials replaced with SpecDics (dictionaries).

    """
    is_special = lambda x: isinstance(x, (partial, Spec)) or callable(x)

    def _asdict(o: object):
        if not isinstance(o, Spec):
            if not isinstance(o, partial):
                o = partial(o)
            o = _compress_partial(o)  # remove nested partials
            o = Spec(o.func, *o.args, **o.keywords)
        args = _tree_map(_asdict, o.args, is_special)
        kwargs = _tree_map(_asdict, o.kwargs, is_special)
        out = dataclasses.replace(o, args=args, kwargs=kwargs).asdict(
            recursive=False, dense=dense
        )
        return out

    return _tree_map(_asdict, o, is_special)


def _tree_map(fn, o: object, is_leaf: Callable[[object], bool]):
    """Apply a function to all elements of a nested object (list, dict, tuple) that satisfy a predicate."""
    if is_leaf(o):
        return fn(o)
    if hasattr(o, "items"):
        # a little hacky, but so that we can use this on ml_collections.ConfigDicts as well
        return type(o)({k: _tree_map(fn, v, is_leaf) for k, v in o.items()})
    elif isinstance(o, (list, tuple)):
        return type(o)(_tree_map(fn, x, is_leaf) for x in o)
    else:
        return o


def _recursive_from_dict(o: object):
    """Recursively creates a Spec from a SpecDict"""
    is_spec_dict = lambda x: _is_spec_dict(x) or _is_dense_spec_dict(x)

    def f(x):
        if _is_dense_spec_dict(x):
            x = _undensify_spec_dict(x)
        args = _tree_map(f, x["args"], is_spec_dict)
        kwargs = _tree_map(f, x["kwargs"], is_spec_dict)
        return Spec(**{**x, "args": args, "kwargs": kwargs})

    return _tree_map(f, o, is_spec_dict)


def _recursive_instantiate_spec(o: object):
    is_spec = lambda x: isinstance(x, Spec)

    def instantiate(x):
        x = dataclasses.replace(
            x,
            args=_tree_map(instantiate, x.args, is_spec),
            kwargs=_tree_map(instantiate, x.kwargs, is_spec),
        )
        return x.instantiate(recursive=False)

    return _tree_map(instantiate, o, is_spec)
