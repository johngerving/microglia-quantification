# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster-rcnn_x101-32x8d_fpn_ms-3x_coco.py'

# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = '/workspace/dataset/'
classes = ('microglia', )
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize', scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')
        )
    )

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        pipeline=test_pipeline,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')
        )
    )

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        pipeline=test_pipeline,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/')
        )
    )

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json')
