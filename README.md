# jax-ml-template
A minimal starter kit for JAX-based machine learning projects.


## Setup

It is suggested to work within a virtual python environment, which can be created using the following commands:

```
python -m venv .mgd-venv
```
Followed by (for linux/macos)
```
source .mgd-venv/bin/activate
```
or (for windows)
```
 .mgd-venv/Scripts/activate.ps1
```

And then (whether you have elected to utilize an virtual environment or not), the following command can be run to prepare your local machine to use this repository:

```
pip install -e .
```

*Good to go!*

## Try it out
### Single Experiments
To simply run a default experiment from the `experiment` directory:
```
python .\train_nn_example.py
```