from itertools import cycle
import copy
from mmengine.runner import Runner
from mmengine.registry import RUNNERS

@RUNNERS.register_module()
class CustomRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_loss_dl = CustomRunner.build_val_loss_dl(self._train_dataloader, self._val_dataloader)

    @staticmethod
    def build_val_loss_dl(train_dataloader, val_dataloader):
        # have to be unitialized
        assert isinstance(train_dataloader, dict)
        assert isinstance(val_dataloader, dict)
        
        # ensure val dataloader for loss calculation uses same sampler/pipeline as train dl (just switch anns and imgs)
        dl = copy.deepcopy(train_dataloader)
        print('DATASET:', dl)
        if 'dataset' in dl:
            if 'ann_file' in dl['dataset']:
                dl['dataset']['ann_file'] = copy.deepcopy(val_dataloader['dataset']['ann_file'])
                dl['dataset']['data_prefix'] = copy.deepcopy(val_dataloader['dataset']['data_prefix'])
            elif 'ann_file' in dl['dataset']['dataset']:
                dl['dataset']['dataset']['ann_file'] = copy.deepcopy(val_dataloader['dataset']['ann_file'])
                dl['dataset']['dataset']['data_prefix'] = copy.deepcopy(val_dataloader['dataset']['data_prefix'])
            else:
                raise Exception("Key 'ann_file' not found")

        dl = CustomRunner.build_dataloader(dl)
        return cycle(dl)
