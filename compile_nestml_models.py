# make the functions available
import sys
from pathlib import Path

from pynestml.frontend.pynestml_frontend import to_nest, install_nest

options = {
    "neuron_parent_class_include": "archiving_node_ext.h",
    "neuron_parent_class": "ArchivingNodeExt",
}

# generate the C++ code:
to_nest(input_path="nestml_models", target_path="module", module_name="nestml_active_dend_module", logging_level="INFO", codegen_opts=options)
#to_nest(input_path="nestml_models", target_path="module", module_name="nestml_active_dend_module", logging_level="INFO")

if len(sys.argv) > 1:
    nest_build_dir = sys.argv[1]
else:
    import nest
    nest_build_dir = str(Path(nest.__path__[0]).parent.parent.parent.parent)
    print("Using {} as the NEST build directory path".format(nest_build_dir))

# compile and install the C++ code:
install_nest("module", nest_build_dir)
