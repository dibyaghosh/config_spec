# config_spec

A small library to help create JSON-serializable configs for functions

## Installation:

```
pip install config_spec 
```

```
pip install git+https://github.com/dibyaghosh/config_spec.git
```

## Basic Usage

```python
from model import create_model
from config_spec import Spec
model_constructor = functools.partial(create_model, num_layers=3)
config = Spec.create(model_constructor) # a dictionary
with open('config.json', 'w') as f:
    json.dump(config, f)
model_constructor = Spec.instantiate(config)
```

```python
from model import create_model
from config_spec import recursive_partial_to_spec, recursive_spec_to_partial
model_constructor = 
config = {
    'model': functools.partial(create_model, num_layers=3),
    'optimizer': functools.partial(torch.optim.Adam, lr=1e-3),
    'num_steps': 1000,
    'batch_size': 32,
} # But this isn't serializable!
config = recursive_partial_to_spec(config) # a dictionary
with open('config.json', 'w') as f:
    json.dump(config, f)

# Later, when you want to load the config:
with open('config.json', 'r') as f:
    config = json.load(f)
config = recursive_spec_to_partial(config)
```


### Notes

This tool was originally created for [Octo](https://github.com/octo-models/octo), a codebase for training robot foundation models.