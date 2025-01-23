from contextlib import nullcontext
import torch
from mmengine.dist import all_reduce_dict
from mmengine.runner.amp import autocast
from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class CustomLoggerHook(Hook):
    priority = 'BELOW_NORMAL'
    def __init__(self, interval: int = 100):
        self.interval = interval
   
    def before_train_iter(self, 
                          runner,
                          batch_idx: int,
                          data_batch = None) -> None:

        # see base class Hook for different ways to check intervals and change to your needs
        if self.every_n_train_iters(runner=runner, n=self.interval):
            outputs = self._get_loss_on_val_batch(runner)  # losses on val data
            all_reduce_dict(outputs, op='mean')
            val_loss = {}
            for k, v in outputs.items():
                val_loss[k] = v.item()  # cuda to cpu
            # do whatever you want with the loss from here, e.g. log it :)
            # or integrated this in a different hook...
            for k, v in val_loss.items():
                runner.message_hub.update_scalar(f'train/{k}_val', v)

    def _get_loss_on_val_batch(self, runner):        
        # we basically run model.train_step but with:
        #
        #     - model.eval() so we don't update batch_norm stats with val data!!!
        #     - torch.no_grad() so we don't produce gradients
        #     - no optim_wrapper.update_params but we have no gradients anyway
        #     - manual amp context instead of optim_wrapper.optim_context() so we don't change anything inside
        #       the optim_wrapper but still get correct amp loss. Not sure if this is actually necessary but
        #       let's not fiddle with optim_wrapper. Maybe we could simply use the context just without calling
        #       optim_wrapper.update_params() but I'm not sure so we just avoid it.
        #
        # see: https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py#L84
        model = runner.model
        if is_model_wrapper(model):  # unwrap DDP
            model = model.module
        data = next(runner.val_loss_dl)
        amp = hasattr(runner.optim_wrapper, 'cast_dtype')
        cast_dtype =  getattr(runner.optim_wrapper, 'cast_dtype', None)
        model.eval()
        with torch.no_grad():
            with autocast(dtype=cast_dtype) if amp else nullcontext():
                data = model.data_preprocessor(data, True)
                losses = model._run_forward(data, mode='loss')  # type: ignore
                parsed_losses, log_vars = model.parse_losses(losses)  # type: ignore
        model.train()
        return log_vars
