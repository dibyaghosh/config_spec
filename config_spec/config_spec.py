from functools import partial
import importlib
from typing import Any, Dict, Tuple, TypedDict, Union


class Spec(TypedDict):
    """A dictionary representation of a function or class with default args and kwargs.
    Useful as a JSON-serializable representation (e.g. for config files or ml_collections) of a callable.

    Note: Spec is just an alias for a dictionary (that is strongly typed), not a real class. So from
    your code's perspective, it is just a dictionary.

    Usage:
        # Create a spec from a callable:
        >>> from my_module import my_function
        # Specs can be created from a partial:
        >>> spec = Spec.create(functools.partial(my_function, arg1=1, arg2=2))
        # or directly from a callable:
        >>> spec = Spec.create(my_function, arg1=1, arg2=2) # or directly
        # Same as above using the fully qualified import string:
        >>> spec = Spec.create("my_module:my_function", arg1=1, arg2=2)

        # Instantiate a callable from a spec:
        >>> Spec.instantiate(spec) == partial(my_function, arg1=1, arg2=2)

    module (str): The module the callable is located in
    name (str): The name of the callable in the module
    args (tuple): The args to pass to the callable
    kwargs (dict): The kwargs to pass to the callable
    """

    module: str
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    auto_call: bool

    @staticmethod
    def create_partial(callable_or_full_name: Union[str, callable], *args, **kwargs) -> "Spec":  # type: ignore
        """Create a module spec from a callable or import string. When this spec is instantiated,
           it will return a partial object.

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

        return Spec(module=module, name=name, args=args, kwargs=kwargs, auto_call=False)

    @staticmethod
    def create_called(callable_or_full_name: Union[str, callable], *args, **kwargs) -> "Spec":  # type: ignore
        """Create a module spec from a callable or import string. When this spec is instantiated,
        it will return the result of calling the callable, with the given args and kwargs.
        """
        spec = Spec.create_partial(callable_or_full_name, *args, **kwargs)
        spec["auto_call"] = True
        return spec

    @staticmethod
    def instantiate(spec: "Spec"):  # type: ignore
        assert is_spec(spec), f"Expected Spec, but got {spec}"
        fn = _import_from_string(spec["module"], spec["name"])
        fn = partial(fn, *spec["args"], **spec["kwargs"])
        if spec["auto_call"]:
            return fn()
        else:
            return fn


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


def is_spec(spec: dict):
    return hasattr(spec, "keys") and set(spec.keys()) == {
        "module",
        "name",
        "args",
        "kwargs",
    }


def _compress_partial(p: partial):
    args = tuple()
    kwargs = dict()
    fn = p.func
    while isinstance(fn, partial):
        args = fn.args + args
        kwargs = {**fn.keywords, **kwargs}
        fn = fn.func
    return partial(fn, *args, **kwargs)


def recursive_create_spec(o: object):
    """Goes through (a potentially nested) object, and converts all partial objects to specs.
    Use this when creating a config file from a partial object.

    Args:
        o: Either a partial object, or a nested (list / dict / tuple) containing partials.
    Returns:
        An object with the same structure as o, but with all partials replaced with Specs (dictionaries).

    """
    if isinstance(o, partial):
        o = _compress_partial(o)  # remove nested partials
        args = recursive_create_spec(o.args)
        kwargs = recursive_create_spec(o.keywords)
        fn = o.func
        return Spec.create_partial(fn, *args, **kwargs)
    elif hasattr(o, "items"):
        # a little hacky, but so that we can use this on ml_collections.ConfigDicts as well
        return type(o)({k: recursive_create_spec(v) for k, v in o.items()})
    elif isinstance(o, (list, tuple)):
        return type(o)(recursive_create_spec(x) for x in o)
    else:
        return o


def recursive_instantiate_spec(o: dict):
    """Goes through (a potentially nested) object, and instantiates all Specs.
    Use this when creating a callable from a config file.

    Args:
        o: Either a Spec object, or a nested (list / dict / tuple) containing Specs.
    Returns:
        An object with the same structure as o, but with all Specs instantiated.
    """
    if is_spec(o):
        args = recursive_instantiate_spec(o["args"])
        kwargs = recursive_instantiate_spec(o["kwargs"])
        return Spec.instantiate(
            {
                "module": o["module"],
                "name": o["name"],
                "args": args,
                "kwargs": kwargs,
                "auto_call": o["auto_call"],
            }
        )
    elif hasattr(o, "items"):
        # a little hacky, but so that we can use this on ml_collections.ConfigDicts as well
        return type(o)({k: recursive_instantiate_spec(v) for k, v in o.items()})
    elif isinstance(o, (list, tuple)):
        return type(o)(recursive_instantiate_spec(x) for x in o)
    else:
        return o
