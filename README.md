# Spiking Temporal Memory

## Overview

This [repository](https://github.com/YounesBouhadjar/SpikingTemporalMemory) contains a detailed mathematical description and a reference implementation of the spiking Temporal Memory (sTM) model, originally developed by [Bouhadjar et al. (2022)][1]. 
...
Various flavors of the model have been used in a number of follow-up studies ([Bouhadjar et al. (2023a)][2], [Bouhadjar et al. (2023b)][3], [Siegel et al. (2023a)][4], [Siegel et al. (2023b)][5], [Bouhadjar et al. (2025)][6]). 
 
## Model description

A detailed mathematical, implementation agnostic description of the model and its parameters is provided [here](docs/ModelDescription_SpikingTemporalMemory.pdf).

## Model implementations
* [PyNEST](PyNEST/README.md)

## Repository contents

|  |  | 
|--|--|
| [`docs`](docs) | model description (implementation agnostic)|
| [`PyNEST`](code) | PyNEST implementaton (python package) |
| &emsp;[`PyNEST/src/spikingtemporalmemory`](PyNEST/src/spikingtemporalmemory) | source code |
| &emsp;[`PyNEST/examples`](PyNEST/examples) | examples illustrating usage of the python package |
| &emsp;[`PyNEST/tests`](PyNEST/tests) | unit tests |


## References

[1]: <https://doi.org/10.1371/journal.pcbi.1010233> "Bouhadjar, Y., Wouters, D. J., Diesmann, M., & Tetzlaff, T. (2022). Sequence learning, prediction, and replay in networks of spiking neurons. PLOS Computational Biology, 18(6), e1010233."
[Bouhadjar, Y., Wouters, D. J., Diesmann, M., & Tetzlaff, T. (2022). Sequence learning, prediction, and replay in networks of spiking neurons. PLOS Computational Biology, 18(6), e1010233.](https://doi.org/10.1371/journal.pcbi.1010233)

[2]: <https://doi.org/10.1371/journal.pcbi.1010989> "Bouhadjar, Y., Wouters, D. J., Diesmann, M., & Tetzlaff, T. (2023). Coherent noise enables probabilistic sequence replay in spiking neuronal networks. PLOS Computational Biology, 19(5), e1010989."
[Bouhadjar, Y., Wouters, D. J., Diesmann, M., & Tetzlaff, T. (2023). Coherent noise enables probabilistic sequence replay in spiking neuronal networks. PLOS Computational Biology, 19(5), e1010989.](https://doi.org/10.1371/journal.pcbi.1010989)

[3]: <https://iopscience.iop.org/article/10.1088/2634-4386/acf1c4> "Bouhadjar, Y., Siegel, S., Tetzlaff, T., Diesmann, M., Waser, R., & Wouters, D. J. (2023). Sequence learning in a spiking neuronal network with memristive synapses. Neuromorphic Computing and Engineering, 3(3), 034014."
[Bouhadjar, Y., Siegel, S., Tetzlaff, T., Diesmann, M., Waser, R., & Wouters, D. J. (2023). Sequence learning in a spiking neuronal network with memristive synapses. Neuromorphic Computing and Engineering, 3(3), 034014.](https://iopscience.iop.org/article/10.1088/2634-4386/acf1c4)

[4]: <https://iopscience.iop.org/article/10.1088/2634-4386/acca45> "Siegel, S., Bouhadjar, Y., Tetzlaff, T., Waser, R., Dittmann, R., & Wouters, D. J. (2023). System model of neuromorphic sequence learning on a memristive crossbar array. Neuromorphic Computing and Engineering, 3(2), 024002."
[Siegel, S., Bouhadjar, Y., Tetzlaff, T., Waser, R., Dittmann, R., & Wouters, D. J. (2023). System model of neuromorphic sequence learning on a memristive crossbar array. Neuromorphic Computing and Engineering, 3(2), 024002.](https://iopscience.iop.org/article/10.1088/2634-4386/acca45)

[5]: <https://doi.org/10.1145/3584954.3585000> "Siegel, S., Ziegler, T., Bouhadjar, Y., Tetzlaff, T., Waser, R., Dittmann, R., & Wouters, D. (2023, April). Demonstration of neuromorphic sequence learning on a memristive array. In Proceedings of the 2023 Annual Neuro-Inspired Computational Elements Conference (pp. 108-114)."
[Siegel, S., Ziegler, T., Bouhadjar, Y., Tetzlaff, T., Waser, R., Dittmann, R., & Wouters, D. (2023, April). Demonstration of neuromorphic sequence learning on a memristive array. In Proceedings of the 2023 Annual Neuro-Inspired Computational Elements Conference (pp. 108-114).](https://doi.org/10.1145/3584954.3585000)

[6]: <> "" 
Bouhadjar Y., Lober M., Neftci E., Diesmann M., Tetzlaff T. (2025). Unsupervised continual learning of complex sequences in spiking neuronal networks. Proceedings of the International Conference on Neuromorphic Systems (ICONS'25)

## Contact
- [Younes Bouhadjar](mailto:y.bouhadjar@fz-juelich.de)
- [Melissa Lober](mailto:m.lober@fz-juelich.de)
- [Tom Tetzlaff](mailto:t.tetzlaff@fz-juelich.de)

## Contribute
We welcome contributions to the documentation and the code. For bug reports, feature requests, documentation improvements, or other issues, please create a [GitHub issue](https://github.com/YounesBouhadjar/SpikingTemporalMemory/issues/new/choose).

## License

The material in this repository is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. For details, see [here](LICENSES/CC-BY-NC-SA-4.0.txt). 
  [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
