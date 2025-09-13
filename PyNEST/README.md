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

- NEST=>3.8.0 ([NEST installation](https://nest-simulator.readthedocs.io/en/stable/installation))
- NESTML>=8.1.0
- matplotlib, numpy, wandb, parameters_space

Build and install the custom neuron and synapse models using NESTML:

```bash
python compile_nestml_models.py
```

## References


License
-------

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This project is licensed under the GNU General Public License v3.0.  
For details, see [here](https://www.gnu.org/licenses/gpl-3.0).
