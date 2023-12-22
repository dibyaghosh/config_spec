# config_spec

This library provides a way of turning a function call into a config (that can be easily modified, extended, serialized to JSON, etc.). 

It may be thought of as a lightweight alternative to `hydra.utils.instantiate` (in theory, it should be compatible with Hydra configs, but I haven't tested it yet).

## TL;DR

```python
>>> from config_spec import Spec
>>> config = dict(Spec(torch.optim.Adam, lr=1e-3))
>>> # same as config = {"_target_": 'torch.optim:Adam', "lr": 0.001}
>>> Spec.instantiate(config) == functools.partial(torch.optim.Adam, lr=1e-3)
```

## Installation:

```
pip install config_spec 
```

## What and Why?



Many ML workflows look like this:
    
```python
# Config (e.g. from JSON or ml_collections or whatever)
config = {
    'learning_rate': 1e-3,
    'num_layers': 3,
    'activations': 'relu',
}

# Then somewhere deep inside our codebase:
tx = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
activations = getattr(torch.nn.functional, config['activations']) # e.g. torch.nn.functional.relu
model = create_model(num_layers=config['num_layers'], activations=activations) 
```

This is fine, but it's not ideal. For one, it's hard to understand exactly what's going on from the config. We now have to look deep into the code to understand the design decisions being made (e.g. what optimizer are we using? Are there any default values that I'm not aware of?) It's also not flexible: 

- What if we want to add new kwargs to `create_model`?
- What if we want to choose the optimizer between Adam or AdamW?
- What if we wanted to use a custom activation function that isn't in `torch.nn.functional`? 

Adding these features require greatly increasing the amount of boilerplate code we have to write in the model init. But it's inherently a problem of configuration -- why should we make our main code more complex? Here's how you can use `config_spec` to solve this problem:

```python
from config_spec import Spec
config = {
    'tx': functools.partial(torch.optim.Adam, lr=1e-3),
    'model': functools.partial(create_model, num_layers=3, activations=torch.nn.functional.relu),
}
# But this isn't easy to configure or to serialize in human-readable format! Enter Spec.asdict()
config = Spec.asdict(config)
# A dictionary that's JSON-serializable and every argument (e.g. which optimizer, activation function, etc.) is specified in the config, and overridable

# Now, inside our codebase:
config = Spec.instantiate(config) # Instantiates all the specs in the dictionary
# config['tx'] == functools.partial(torch.optim.Adam, lr=1e-3)
tx = config['tx'](model.parameters()) 
# config['model'] == functools.partial(create_model, num_layers=3, activations=torch.nn.functional.relu)
model = config['model']() 
```

Spec.asdict() converts our config into the following friendly dictionary (if you want, you can also just create this dictionary directly):

```python
config = {
    "tx": {
        "_target_": 'torch.optim.adam:Adam',
        "lr": 0.001,
    },
    "model": {
        "_target_": 'model:create_model',
        "num_layers": 3,
        "activations": {"_target_": 'torch.nn.functional:relu',},
    },
}
```

How is this better? 1) It makes the config more transparent (e.g. we see exactly what changing the config does) 2) It makes things more easy to override

```python
    # We can easily modify the config dict in any way we want
    >>> config['tx']['lr'] = 1e-4
    >>> config['tx']['_target_'] = 'torch.optim:AdamW'
    >>> config['tx']['beta1'] = 0.9 # Add new kwargs easy!
    >>> config['model']['activations']['_target_'] = 'torch.nn.functional:gelu'
```


## Basic Usage

You can create a spec either by using the `Spec` class, or just directly creating a dictionary with (at least) a `_target_` key. The `_target_` key is a fully qualified import name (e.g. `torch.optim:Adam`).

```python
>>> spec = Spec(torch.optim.Adam, lr=1e-3)
<Spec: functools.partial(torch.optim.adam:Adam, lr=0.001)>
>>> d = dict(spec)
{'_target_': 'torch.optim.adam:Adam', 'lr': 0.001}
>>> Spec.instantiate(spec) == Spec.instantiate(d) == functools.partial(torch.optim.Adam, lr=1e-3)
```

```python
from model import create_model
from config_spec import Spec
config = {
    'model': functools.partial(create_model, num_layers=3),
    'optimizer': functools.partial(torch.optim.Adam, lr=1e-3),
    'num_steps': 1000,
    'batch_size': 32,
} # But this isn't serializable!
config = Spec.asdict(config) # a dictionary
with open('config.json', 'w') as f:
    json.dump(config, f)

# Later, when you want to load the config:
with open('config.json', 'r') as f:
    config = json.load(f)
config = Spec.instantiate(config) # get back the original config
```


### Notes

This tool was originally created for [Octo](https://github.com/octo-models/octo), a codebase for training robot foundation models. I really hadn't realized that Hydra did exactly the same thing until I was almost done with this library. I decided to publish it anyway because I think it's a useful tool, and it's a lot simpler than Hydra + it works with other libraries like ml_collections.