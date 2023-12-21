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
        >>> spec = Spec(torch.optim.Adam, lr=1e-3)
        >>> Spec.instantiate(spec) == functools.partial(torch.optim.Adam, lr=1e-3)

        # Specs can be easily serialized and de-serialized
        >>> d = dict(spec)
        {'target': 'torch.optim:Adam', 'lr': 1e-3, '__config_spec__': True}
        >>> d['lr'] = 1e-4
        >>> Spec.instantiate(d) == functools.partial(torch.optim.Adam, lr=1e-4)

        # Supports calling functions too and nested specs
        >>> lr_spec = Spec(torch.square)(1e-3) # calling a spec will cause it to be called when instantiated
        >>> spec = Spec(torch.optim.Adam, lr=lr_spec)
        >>> Spec.instantiate(spec) == functools.partial(torch.optim.Adam, lr=1e-6)


    target (str): The fully-qualified name of a callable (e.g. "torch.optim:Adam")
    args (tuple): The args to pass to the callable
    kwargs (dict): The kwargs to pass to the callable
    return_type (str): Whether to return a functools.partial object ('partial') or to call it with the provided params ('called')
    """

    target: str
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    return_type: str = "partial"

    @classmethod
    def instantiate(cls, o: object):
        """Instantiates a spec by importing the target function and partial/calling the callable with the given args and kwargs.
            Will recursively instantiate any nested specs as well.

        Args:
            o: Either a Spec, dictionary representation of a Spec, or a nested (list / dict / tuple) structure containing Specs.
        Returns:
            The instantiated object (if o is a Spec / dict), or a nested structure containing instantiated objects (if o is a nested structure containing Specs)
        """
        # Convert all dictionary representations of Specs to Specs
        o = cls.from_dict(o)
        is_spec = lambda x: isinstance(x, cls)

        def instantiate(x: Spec):
            x.args = _tree_map(instantiate, x.args, is_spec)
            x.kwargs = _tree_map(instantiate, x.kwargs, is_spec)
            return x._instantiate()

        return _tree_map(instantiate, o, is_spec)

    def _instantiate(self):
        """Shallowly instantiate the spec, by importing the module and partial/calling the callable with the given args and kwargs.

        You should probably use Spec.instantiate instead, which will recursively instantiate any nested specs as well.
        """
        fn = _import_from_string(self.target)
        if len(self.args) > 0 or len(self.kwargs) > 0:
            fn = partial(fn, *self.args, **self.kwargs)
        if self.return_type == "called":
            return fn()
        elif self.return_type == "partial":
            return fn
        else:
            raise ValueError(f"Invalid return_type: {self.return_type}")

    def __init__(self, target: Union[str, callable], *args, **kwargs):
        """Create a spec from a callable. Note: you can also do the standard dataclass init

        Args:
            target (object): A Callable
            args (tuple, optional): Passed into callable upon instantiation.
            kwargs (dict, optional): Passed into callable upon instantiation.
        """

        if isinstance(target, str):
            # Do the dataclass standard init
            self._standard_init(target, *args, **kwargs)
        else:
            self._init_from_callable(target, *args, **kwargs)

    def __post_init__(self):
        assert (
            self.target.count(":") == 1
        ), f"target must be a fully qualified import string (e.g. 'torch.optim:Adam'), received {self.target}"

    @classmethod
    def from_dict(cls, o: object):
        """Recurses through a nested (list / dict / tuple), converting all SpecDicts to Specs."""
        is_spec_dict = lambda x: _is_spec_dict(x) or _is_dense_spec_dict(x)

        def f(x: dict):
            if _is_dense_spec_dict(x):
                x = _undensify_spec_dict(x)

            args = _tree_map(f, x["args"], is_spec_dict)
            kwargs = _tree_map(f, x["kwargs"], is_spec_dict)
            return cls(**{**x, "args": args, "kwargs": kwargs})

        return _tree_map(f, o, is_spec_dict)

    def _standard_init(self, *args, **kwargs):
        """Standard dataclass init, but with some extra checks and modifications."""
        if kwargs.get("__config_spec__", False):
            kwargs = _undensify_spec_dict(kwargs)

        # Standard dataclass init part
        for n, field in enumerate(dataclasses.fields(self)):
            value = (
                args[n] if n < len(args) else kwargs.get(field.name, _default(field))
            )
            assert value != dataclasses.MISSING, f"Missing argument {field.name}"
            setattr(self, field.name, value)

        # Recursively convert all kwargs and args to specs
        def convert_to_spec(x):
            if hasattr(x, "keys"):  # is a dict
                return Spec(**x)
            elif not isinstance(x, Spec):  # is a callable
                return Spec(x)
            return x  # is already a spec

        self.args = _tree_map(
            convert_to_spec,
            self.args,
            lambda x: callable(x) or _is_spec_dict(x) or _is_dense_spec_dict(x),
        )
        self.kwargs = _tree_map(
            convert_to_spec,
            self.kwargs,
            lambda x: callable(x) or _is_spec_dict(x) or _is_dense_spec_dict(x),
        )

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
            self.return_type != "called"
        ), "Cannot call partial on a spec that has already been called"
        new_args = self.args + args
        new_kwargs = {**self.kwargs, **kwargs}
        return dataclasses.replace(self, args=new_args, kwargs=new_kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Add additional args and kwargs to the spec, and marks it for calling. (Functional, not in-place)

        When this spec is instantiated, it will execute the callable with the given args and kwargs, instead of returning a partial object.
        """
        assert (
            self.return_type != "called"
        ), f"Cannot call a spec that has already been called: {self!r}"
        new_args = self.args + args
        new_kwargs = {**self.kwargs, **kwds}
        return dataclasses.replace(
            self, args=new_args, kwargs=new_kwargs, return_type="called"
        )

    @classmethod
    def asdict(cls, o: object):
        """Converts a Spec to a SpecDict, recursively. Use this when serializing a config."""

        def f(x: Callable):
            print(x)
            if not isinstance(x, cls):
                x = cls(x)
            x: Spec = dataclasses.replace(x)  # Copy, don't mutate the original
            x.args = cls.asdict(x.args)
            x.kwargs = cls.asdict(x.kwargs)
            return _densify_spec_dict(dataclasses.asdict(x))

        return _tree_map(f, o, callable)

    def __iter__(self):
        # so that we can do dict(spec)
        return iter(self.asdict(self).items())

    def __repr__(self) -> str:
        args = [repr(x) for x in self.args]
        args.extend(f"{k}={v!r}" for (k, v) in self.kwargs.items())
        if self.return_type != "called":
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


def _default(field: dataclasses.Field):
    if field.default != dataclasses.MISSING:
        return field.default
    elif field.default_factory != dataclasses.MISSING:
        return field.default_factory()
    else:
        return dataclasses.MISSING


def _densify_spec_dict(spec_dict: dict):
    """Converts a spec dict to a denser form, by
    1) flattening kwargs into the dict
    2) removing default options (e.g. args=(), kwargs={}, return_type='partial')

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
    for field in dataclasses.fields(Spec):
        default_value = _default(field)
        if spec_dict[field.name] == default_value:
            spec_dict.pop(field.name)

    spec_dict["__config_spec__"] = True
    return spec_dict


def _undensify_spec_dict(spec_dict: dict):
    """Reverts a densified spec dict back to its original form"""
    spec_dict = spec_dict.copy()
    assert spec_dict.pop("__config_spec__", False), "Spec dict is not dense"

    for field in dataclasses.fields(Spec):
        if field.name not in spec_dict:
            spec_dict[field.name] = _default(field)

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
