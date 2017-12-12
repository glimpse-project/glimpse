# Glimpse Python bindings

## Building

It's recommended to run this inside a virtualenv, so as not to pollute the
system Python package index.

Build the glimpse module by running the following:

```
make
pip install --user --force-reinstall dist/glimpse*.whl
```

The 'glimpse' module will now be available in Python. Make sure that 
libglimpse-python-tools.so is available in the library path before importing.
