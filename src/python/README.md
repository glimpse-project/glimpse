# Glimpse Python bindings

## Building

It's recommended to run this inside a virtualenv, so as not to pollute the 
system Python package index. First, ensure libglimpse.so is built in the 
parent directory and that SWIG is installed and available in the PATH. Then, 
run the following:

```
make
pip install dist/glimpse*.whl
```

The 'glimpse' module will now be available in Python. Make sure that 
libglimpse.so is available in the library path before importing.
