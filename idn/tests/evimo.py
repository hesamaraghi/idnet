from torch.utils.data import DataLoader

from .test import Test
from ..loader.loader_dsec import train_collate
from ..loader.loader_evimo import assemble_evimo_sequences


class TestEVIMO2V2(Test):
    def configure_dataloader(self):
        split = self.spec.dataset.val.get(
            "split", self.spec.dataset.get("split", "eval")
        )
        valid_set = assemble_evimo_sequences(
            self.spec.dataset.common.data_root,
            split=split,
            include_seq=self.spec.dataset.val.get("seq", None),
            config=self.spec.dataset,
            num_bins=self.spec.dataset.get("num_voxel_bins", None),
        )
        assert self.spec.data_loader.args.shuffle is False, (
            "shuffle must be false for val run."
        )
        return [
            DataLoader(seq, collate_fn=train_collate, **self.spec.data_loader.args)
            for seq in valid_set
        ]
