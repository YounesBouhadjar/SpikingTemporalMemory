# Extension of the spiking TM network

This repository is organized as follows:

- `experiments` contains simulation scripts
- `figures` contains scripts to produce figures
- `data` contains simulation data
- `nestml_script` contains custom neuron model developed using NESTML

For questions or feedback please contact Younes Bouhadjar: y.bouhadjar@fz-juelich.de

# Software dependencies

To meet the software requirements, create and activate the conda environment ```spiking-htm```:
```bash
mamba env create -f environment.yaml #TODO outdated environment see below
mamba activate ext_spiking-tm
```

In addition, `pdflatex` needs to be available for the generation of some of the figures.

# Reproducing simulation data

The code relies on a custom synapse and neuron model that you can get by installing these NEST and NESTML versions:

## Preparing NEST installation

1. Preferably, it is better to use `mamba` instead of `conda`, as it is much faster at dealing with large files.

    * Install `mamba`:

   ```
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

   * in case you have been using conda, remove conda base
   ```
   conda config --set auto_activate_base false
   ```
   and launch a new terminal

2. Create a virtual environment with the necessary packages:

   ```
   mamba create --name ext_spiking-tm python==3.9
   mamba activate ext_spiking-tm
   mamba install -c conda-forge nest-simulator boost ipython cxx-compiler cmake gsl
   pip install nestml
   ```
   nestml may require pygsl, but notice version conflict between pygsl and gsl
   
3. edit nest-config (according to [nestml doc](https://nestml.readthedocs.io/en/latest/installation.html#anaconda-installation))

4. Build and install the custom neuron and synapse models using NESTML:
   ```bash
    cd ../ 
    python compile_nestml_models.py
    ```

## Running the hyperparameter search

* To run the hyperparameter search you need to do the following:

    * Add your config file to wb_configs, see wb_configs/default_sequence_prediction.yaml for an example
    * Check `training.py`, whether you are properly using the parameters you want to do hyperparameter search over: use either the parser or the dict `wandb.config` 
    * Run: python generate_sweep_id.py wb_configs/xxx_config.yaml, this would give you a sweep_id
    * Run: wandb agent --count 500 wandb-user_name/project_name/sweep_id. This run the training sequentially for 500 times
    * You could run the above command multiple times for a faster parameter search: you could do it manually or run `submission_cluster_sweep.py wandb-user_name/project_name/sweep_id 20`, where 20 is the number of parallel runs
