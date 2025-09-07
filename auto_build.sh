#!/bin/bash/
conda activate spiking-htm
mkdir -p $PWD/build
cd $PWD/build
 
cmake -DCMAKE_INSTALL_PREFIX=$PWD/ -Dwith-gsl=$CONDA_PREFIX -DREADLINE_ROOT_DIR=$CONDA_PREFIX -DLTDL_ROOT_DIR=$CONDA_PREFIX -Dwith-openmp=ON -Dwith-mpi=OFF ../../ && make -j8 && make install
