# config_spec

A small library to help create JSON-serializable configs for functions. 

## What and Why?

This library provides a way of turning a function call into a config (that can be easily modified, extended, serialized to JSON, etc.)

Many ML workflows look like this:
    
```python
    # Config (e.g. from JSON or ml_collections or Hydra, whatever)
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

Adding these features require greatly increasing the amount of boilerplate code we have to write in the model init. But it's inherently a problem of configuration! Here's how you can use `config_spec` to solve this problem:

```python
from config_spec import Spec
config = Spec.asdict({
    'tx': functools.partial(torch.optim.Adam, lr=1e-3),
    'model': functools.partial(create_model, num_layers=3, activations=torch.nn.functional.relu),
})
# A dictionary that's fully JSON-serializable! And every argument (e.g. which optimizer, activation function, etc.) is specified in the config, and overridable


# Now, somewhere deep inside our codebase:
config = Spec.instantiate(config) # Instantiates all the specs in the dictionary
# config['tx'] == functools.partial(torch.optim.Adam, lr=1e-3)
tx = config['tx'](model.parameters()) 
# config['model'] == functools.partial(create_model, num_layers=3, activations=torch.nn.functional.relu)
model = config['model']() 
```

How is this better? 1) It makes the config more transparent (e.g. we see exactly what changing the config does) 2) It makes things more easy to override

```python
    # We can easily modify the config dict in any way we want
    >>> config['tx']['lr'] = 1e-4
    >>> config['tx']['target'] = 'torch.optim:AdamW
    >>> config['tx']['beta1'] = 0.9 # Add new kwargs easy!
    >>> config['model']['activations']['target'] = 'torch.nn.functional:gelu'
```

What does this config dictionary look like anyways?

```python
>>> config
{
    tx: {
        target: 'torch.optim.adam:Adam',
        lr: 0.001,
        __config_spec__: True,
    },
    model: {
        target: 'model:create_model',
        num_layers: 3,
        activations: {
            target: 'torch.nn.functional:relu',
            __config_spec__: True,
        },
        __config_spec__: True,
    },
}
```

## Installation:

```
pip install config_spec 
```

```
pip install git+https://github.com/dibyaghosh/config_spec.git
```

## Basic Usage

You can create a spec either by using the `Spec` class, or just directly creating a dictionary with (at least) a `target` key, and a `__config_spec__` key. The `target` key is a fully qualified import name (e.g. `torch.optim.Adam`), and the `__config_spec__` key is to help the library know that this dictionary is a spec, and not just a regular dictionary.

```python
>>> spec = Spec(torch.optim.Adam, lr=1e-3)
<Spec: functools.partial(torch.optim.adam:Adam, lr=0.0001)>
>>> d = dict(spec)
{
    'target': "torch.optim.adam:Adam" # A fully qualified import name
    'lr': 1e-3,
    '__config_spec__': True # Tells the library that this is a spec, and not just a regular dictionary
}
>>> spec.instantiate() ==  Spec.instantiate_from_dict(d) == functools.partial(torch.optim.Adam, lr=1e-3)
```

```python
from model import create_model
from config_spec import Spec, asdict
config = {
    'model': functools.partial(create_model, num_layers=3),
    'optimizer': functools.partial(torch.optim.Adam, lr=1e-3),
    'num_steps': 1000,
    'batch_size': 32,
} # But this isn't serializable!
config = asdict(config) # a dictionary
with open('config.json', 'w') as f:
    json.dump(config, f)

# Later, when you want to load the config:
with open('config.json', 'r') as f:
    config = json.load(f)
config = Spec.instantiate_from_dict(config)
```


### Notes

This tool was originally created for [Octo](https://github.com/octo-models/octo), a codebase for training robot foundation models.