# ranking-utils
This repository provides miscellaneous utilities for ranking models, such as:
* Dataset pre-processing
* Training and evaluation

## Installation
Clone this repository and run:
```bash
python -m pip install .
```

## Dataset Pre-Processing
The dataset pre-processing script reads a dataset and creates training, dev and test sets (hdf5 format) that can be used by the ranking models. Run the script as follows to see available options:
```bash
python -m ranking_utils.scripts.create_h5_data
```
The following datasets are currently supported:
* [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [InsuranceQA (v2)](https://github.com/shuzi/insuranceQA)
* [MS MARCO (v1) passage and document ranking (TREC-DL test sets)](https://microsoft.github.io/msmarco/)
* Any dataset in generic TREC format

The script uses [Hydra](https://hydra.cc). Refer to the documentation for detailed instructions on how to configure arguments. 

### Example
The following pre-processes the ANTIQUE dataset:
```bash
python -m ranking_utils.scripts.create_h5_data \
    dataset=antique \
    dataset.root_dir=/path/to/antique/files \
    hydra.run.dir=/path/to/output/files
```

In order to see all available options for a dataset, run:
```bash
python -m ranking_utils.scripts.create_h5_data \
    dataset=antique \
    --help
```
