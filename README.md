# ranking-utils
This repository provides miscellaneous utilities for ranking models.

## Installation
Clone this repository and run:
```
python -m pip install -e .
```

## Usage

### Dataset Pre-Processing
The dataset pre-processing script reads a dataset and creates training, dev and test sets (hdf5 format) that can be used by the ranking models. Run the script as follows to see available options:
```
python -m ranking_utils.scripts.create_h5_data
```
The following datasets are currently supported:
* [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/)
* [FiQA Task 2](https://sites.google.com/view/fiqa/home)
* [InsuranceQA (v2)](https://github.com/shuzi/insuranceQA)
* [MS MARCO (v1) passage and document ranking (TREC-DL test sets)](https://microsoft.github.io/msmarco/)
* Any dataset in generic TREC format

The script uses [Hydra](https://hydra.cc). Refer to the documentation for detailed instructions on how to configure arguments. 

#### Example
The following pre-processes the ANTIQUE dataset:
```
python -m ranking_utils.scripts.create_h5_data \
    dataset=antique \
    dataset.root_dir=/path/to/antique/files \
    hydra.run.dir=/path/to/output/files
```

In order to see all available options for a dataset, run:
```
python -m ranking_utils.scripts.create_h5_data \
    dataset=antique \
    --help
```

### Ranking
Implementing a ranker requires two components:
1. A DataProcessor (specific to your model) subclasses `ranking_utils.model.data.DataProcessor` and implements the following methods:
    - `get_model_input(self, query: str, doc: str) -> ModelInput`: Transforms a query-document pair into an input that is suitable for the model.
    - `get_model_batch(self, inputs: Iterable[ModelInput]) -> ModelBatch`: Creates a model batch from multiple inputs.
2. The ranking model itself subclasses `ranking_utils.model.Ranker` and implements the following methods:
    - `forward(self, batch: ModelBatch) -> torch.Tensor`: Computes query-document scores, output shape `(batch_size, 1)`
    - `configure_optimizers(self) -> Tuple[List[Any], List[Any]]`: Configures optimizers (and schedulers). Refer to the [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers).

You can then train your model using the usual PyTorch Lightning setup. For example:
```python
from pathlib import Path
from pytorch_lightning import Trainer
from ranking_utils.model.data import H5DataModule
from ranking_utils.model import TrainingMode
from my_ranker import MyRanker, MyDataProcessor

data_module = H5DataModule(
    data_processor=MyDataProcessor(...),
    data_dir=Path(...),
    fold_name="fold_0",
    batch_size=32
)
model = MyRanker(...)
data_module.training_mode = model.training_mode = TrainingMode.PAIRWISE
model.pairwise_loss_margin = 0.2
Trainer(...).fit(model=model, datamodule=data_module)
```

#### Validation
After each epoch, the ranker automatically computes the following ranking metrics on the validation set:
* `val_RetrievalMAP`: mean average precision
* `val_RetrievalMRR`: mean reciprocal rank
* `val_RetrievalNormalizedDCG`: nDCG score

These can be used in combination with callbacks, e.g. [early stopping](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.EarlyStopping.html?highlight=earlystopping).

#### Examples
Example implementations of various models using this library can be found [here](https://github.com/mrjleo/ranking-models).
