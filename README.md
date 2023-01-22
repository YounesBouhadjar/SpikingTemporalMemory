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
conda env create -f environment.yaml
conda activate spiking-htm
```

In addition, `pdflatex` needs to be available for the generation of some of the figures.

# Reproducing simulation data

The code relies on a custom synapse and neuron model that you can get by installing these specific NEST and NESTML versions:

## Preparing NEST installation

1. Get the following NEST version [git repository](https://github.com/YounesBouhadjar/nest-simulator/tree/stdsp_synapse) and checkout the correct branch:
   ```bash
   git clone git@github.com:YounesBouhadjar/nest-simulator.git nest-shtm
   cd nest-shtm
   git checkout stdsp_synapse
   ```
   the only difference between this NEST version and the master version is that it includes the new synapse model 'stdsp_synapse'

2. Build NEST from the modified source: 
   
   * Make sure that the conda environment `spiking-htm` is activated
   * Make sure that you have installed:
     * Cython 0.28.3 or higher
     * CMake 3.12 or higher
     * these requirements should be already defined in `environment.yaml`

   * For your convenience we provide a script that builds the NEST code:
     * copy `auto_build.sh` to the `nest-shtm` directory:
     ```bash
     cp ../auto_build.sh .
     ```
     and then run
     ```bash
     bash auto_build.sh
     ```

     * executing this creates a build directory in `nest-shtm` under the name: `build/stdsp_synapse`
   
   * source the `nest_vars.sh` script:
     ```bash
     source ./build/stdsp_synapse/bin/nest_vars.sh
     ```
 
     * you could add `source <nest_dir>/nest-shtm/build/stdsp_synapse/bin/nest_vars.sh` to you `.bashrc`, 
       this way the environment variables are set automatically whenever you open a new terminal
     * alternatively, you could add this to `env_vars.sh` in your conda environment:
     ```bash
     cd $CONDA_PREFIX
     mkdir -p ./etc/conda/activate.d
     touch ./etc/conda/activate.d/env_vars.sh
     echo "source <nest_dir>/nest-shtm/build/stdsp_synapse/bin/nest_vars.sh" > ./etc/conda/activate.d/env_vars.sh
     ```
     :warning: don't forget to change <nest_dir>   
 
   * For more information look at the [NEST installation instructions](https://nest-simulator.readthedocs.io/en/stable/installation/index.html#advanced-install).

3. Get the following NESTML version [git repository](https://github.com/YounesBouhadjar/nestml)
   ```bash
   git clone git@github.com:YounesBouhadjar/nestml.git
   cd nestml
   git checkout iaf_psc_exp_nonlineardendrite
   ```
   This version of NESTML contains custom code for the neuron model defined in `nestml_models/iaf_psc_exp_nonlineardendrite.nestml`
4. Install NESTML: 
   ```bash     
   python setup.py install --user  
   ```

5. Build and install the custom neuron model using NESTML:
   ```bash
    cd ../ 
    python compile_nestml_models.py
    ```

