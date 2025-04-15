from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmdet.apis import inference_detector, init_detector

import os
from pycocotools.coco import COCO
from torchmetrics.classification import BinaryJaccardIndex

@HOOKS.register_module()
class FindIoU(Hook):
    def __init__(self, name):
        os.makedirs("bestepochs", exist_ok=True)
        # Some Necessary Variables for me
        self.bestIoU = 0
        self.bestepoch = None
        self.name = name
        self.metric = BinaryJaccardIndex()
        
    def after_val(self, runner, **kwargs):
        IoUs = []
        # TO LOAD THE MODEL FROM THE RECENT WEIGHT FILE
        checkpoint_file = runner.work_dir + f"/epoch_{runner.epoch}.pth"
        model = init_detector(runner.cfg, checkpoint_file, device='cuda:0')
        meanIoU = []
        val_file = runner.cfg.val_dataloader.dataset.data_root + runner.cfg.val_dataloader.dataset.ann_file
        test_file = runner.cfg.test_dataloader.dataset.data_root + runner.cfg.test_dataloader.dataset.ann_file
        for f_type, json_path in zip(['pGen1', 'pGen2'], [val_file, test_file]):
            
            # json_path = f"{data_type}.json"
            coco = COCO(json_path)
            img_dir = f"combined_data"
            cat_ids = coco.getCatIds()
            for idx, img_data in coco.imgs.items():
                anns_ids = coco.getAnnIds(imgIds=img_data['id'], catIds=cat_ids, iscrowd=None)
                anns = coco.loadAnns(anns_ids)

                print(anns)
                truth_mask = coco.annToMask(anns[0])
                for i in range(1,len(anns)):
                    truth_mask = np.maximum(truth_mask,coco.annToMask(anns[i])*1)

                img = f'combined_data/{img_data["file_name"]}'  # or img = mmcv.imread(img), which will only load it once
                # PERFORMING INFERENCE
                result = inference_detector(model, img)
                # outputs = predictor(im)

                pred_mask = np.zeros_like(truth_mask)
                for i in result.pred_instances.masks.type(torch.int8):
                    pred_mask = np.maximum(pred_mask, i.to('cpu').numpy().astype(np.uint8))
                    
                # frame = label2rgb(pred_mask, cv2.imread(img), alpha=0.3, bg_label=0)*255
    
                target = torch.tensor(truth_mask)
                preds = torch.tensor(pred_mask)
            
                intersection_mask = np.logical_and(pred_mask == 1, truth_mask == 1)
                pred_mask[truth_mask == 1] = 2
                pred_mask[intersection_mask] = 3
                # Repeating Channels to make it three channels
                pred_mask = np.tile(pred_mask[..., np.newaxis], (1,1,3))

                IoUs.append(self.metric(preds, target).item())
                
            # Collect all meanIoUs for all Generalization Patients
            meanIoU.append(sum(IoUs)/len(IoUs))
            print(f"IoU: {sum(IoUs)/len(IoUs)}")
            
        for IoU, log in zip(meanIoU, ['pGen1', 'pGen2']):
            # wandb.log({f'coco/{log}':IoU, 'coco/epoch':runner.epoch})
            print({f'coco/{log}':IoU, 'coco/epoch':runner.epoch})
            
        meanIoU = sum(meanIoU)/len(meanIoU)
        if meanIoU > self.bestIoU:
            self.bestIoU = meanIoU
            self.bestepoch = checkpoint_file

        print(f"meanIoU: {meanIoU}")
        # wandb.log({'coco/iou':meanIoU, 'coco/epoch':runner.epoch})
        print({'coco/iou':meanIoU, 'coco/epoch':runner.epoch})
