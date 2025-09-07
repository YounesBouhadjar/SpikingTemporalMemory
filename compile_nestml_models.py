# make the functions available
from pynestml.frontend.pynestml_frontend import generate_nest_target
import re
import nest
from pathlib import Path

nestml_neuron_model = 'iaf_psc_exp_nonlineardendrite_neuron'
nestml_synapse_model = 'stdsp_synapse'

nest_build_dir = str(Path(nest.__path__[0]).parent.parent.parent.parent)
input_path = 'nestml_models/'

def compile_neuron_model():
  # compile nestml neuron model
  generate_nest_target(input_path=input_path + nestml_neuron_model + ".nestml",
                       target_path="module",
                       logging_level="ERROR",
                       module_name="nestml_"+ nestml_neuron_model + "_module",
                       codegen_opts={"nest_path":  nest_build_dir})

  nest.Install("nestml_"+ nestml_neuron_model + "_module")


def compile(nestml_neuron_model_name, nestml_synapse_model_name):

  # generate the code for neuron and synapse (co-generated)
  generate_nest_target(input_path=[input_path + nestml_neuron_model + ".nestml",
                                   input_path + nestml_synapse_model + ".nestml"],
                       target_path="module",
                       logging_level="ERROR",
                       module_name="nestml_" + nestml_neuron_model + "_" + nestml_synapse_model + "_module",
                       suffix="_nestml",
                       codegen_opts={"nest_path": nest_build_dir,
                                     #"neuron_parent_class": "StructuralPlasticityNode",
                                     #"neuron_parent_class_include": "structural_plasticity_node.h",
                                     "neuron_synapse_pairs": [{"neuron": nestml_neuron_model_name,
                                                                 "synapse": nestml_synapse_model_name,
                                                                 "post_ports": ["post_spikes", ["z_post", "z"]]}],
                                     "delay_variable": {"stdsp_synapse": "d"},
                                     "weight_variable": {"stdsp_synapse": "w"}})
                                                                 

compile_neuron_model()
compile(nestml_neuron_model, nestml_synapse_model)
