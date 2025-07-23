# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster-rcnn_x101-32x8d_fpn_ms-3x_coco.py'

interval = 1
max_keep_ckpts = 1
save_best = 'coco/bbox_mAP'

default_hooks = dict(
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=10,
        min_delta=0.005),
    checkpoint=dict(
        type="CheckpointHook",
        interval=interval,
        save_begin=100,
        max_keep_ckpts=max_keep_ckpts,
        save_best=save_best)
)

dataset_type = 'CocoDataset'
data_root = '/workspace/dataset/'
classes = ('activated', 'non-activated')
num_classes = len(classes)
backend_args = None

metainfo=dict(classes=classes, palette=[200,20,60])

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'))))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))

test_evaluator = dict(
        ann_file=data_root + 'test/_annotations.coco.json')
val_evaluator = dict(
        ann_file=data_root + 'valid/_annotations.coco.json')


# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes)))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=75, val_interval=1)
