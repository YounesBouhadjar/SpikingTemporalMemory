# PyNEST implementation

## Installing the python package `spikingtemporalmemory`

The PyNEST implementation of the model is provided in the form of a python package `spikingtemporalmemory` and is contained in `SpikingTemporalMemory/PyNEST`:
  
  ```bash
  cd PyNEST
  ```

  We recommend installing the python package inside a python environment:
- Create a python environment
  ```bash
  python -m venv spikingtm
  ```
- Activate the python environment:
  ```
  source spikingtm/bin/activate
  ```
- Note: NEST needs to be installed locally in the virtual environment (see software requirements)

The `spikingtemporalmemory` python package can be installed using:
  ```bash
  pip install -U pip
  pip install .
  ```

## Testing

Executing
```bash
pytest
```
runs the unit test(s) in `SpikingTemporalMemory/PyNEST/tests`.

## Usage

After installation, the `spikingtemporalmemory` python package can be imported in a python application using

```python
import spikingtemporalmemory
```

See [this example]() for a more detailed illustration of how the package can be used.


## Software requirements

- NEST ([NEST installation](https://nest-simulator.readthedocs.io/en/stable/installation))
- Python 3.12.3
- matplotlib, numpy, wandb, nestml

## References


License
-------
