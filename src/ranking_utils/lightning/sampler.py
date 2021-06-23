from typing import Iterable
from torch.utils.data.distributed import DistributedSampler

from ranking_utils.lightning.datasets import ValTestDatasetBase


class DistributedQuerySampler(DistributedSampler):
    """A distributed sampler that samples queries instead of indices.
    This is required so that ranking measures (e.g. AP) can be computed on each machine.

    Args:
        dataset (ValTestDatasetBase): The validation dataset
    """
    def __init__(self, dataset: ValTestDatasetBase):
        super().__init__(dataset, None, None, False, 0)

        # first, select the query IDs (i.e. corresponding offsets) for this rank
        num_q_ids = len(self.dataset.offsets)
        offsets_i1 = int(self.rank / self.num_replicas * num_q_ids)
        offsets_i2 = int((self.rank + 1) / self.num_replicas * num_q_ids)

        # next, select the corresponding dataset indices
        indices_start = self.dataset.offsets[offsets_i1]

        # in this case (last machine) there is no final offset, so we use the dataset length
        if self.rank == self.num_replicas - 1:
            indices_stop = len(dataset)
        else:
            indices_stop = self.dataset.offsets[offsets_i2]
        self.indices = list(range(indices_start, indices_stop))

    def __iter__(self) -> Iterable[int]:
        """Yield all indices corresponding to the queries.

        Yields:
            int: A single item index
        """
        return iter(self.indices)

    def __len__(self) -> int:
        """Return the number of items for this rank.

        Returns:
            int: The number of items
        """
        return len(self.indices)
