import numpy as np
from random import shuffle
from sklearn.model_selection import KFold
import argparse
from pathlib import Path
import json
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann-file',
        type=str,
        help='An existing annotation file for a COCO dataset.',
        default='./data/coco/')
    parser.add_argument(
        '--fold',
        type=int,
        help='K-fold cross validation for semi-supervised object detection.',
        default=5)
    args = parser.parse_args()
    return args

def split_coco(ann_file: str, n_folds: int):
    ann_file = Path(ann_file) 
    if not ann_file.is_file():
        raise Exception(f'File {ann_file} not found')

    with open(ann_file) as f:
        obj = json.load(f)

        for el in ['images', 'annotations']:
            if el not in obj:
                raise Exception(f'Field "{el}" not found in {ann_file}')
            if not isinstance(obj[el], list):
                raise Exception(f'Field "{el}" must be a list')
    
    images = np.array(obj['images'])
    annotations = np.array(obj['annotations'])

    shuffle(images)

    kf = KFold(n_splits=n_folds)

    train_folds = []
    test_folds = []

    fold = 1
    for train_index, test_index in kf.split(images):
        train_images = images[train_index]
        test_images = images[test_index]

        train_annotations = []
        for image in train_images:
            image_id = image['id']
            train_annotations = train_annotations + [annotation for annotation in annotations if annotation['image_id'] == image_id]

        test_annotations = []
        for image in test_images:
            image_id = image['id']
            test_annotations = test_annotations + [annotation for annotation in annotations if annotation['image_id'] == image_id]

        assert len(train_annotations) + len(test_annotations) == len(annotations), 'Length of full annotations set should be equal to the sum of train and test annotations for fold'

        train_obj = copy.deepcopy(obj)
        test_obj = copy.deepcopy(obj)

        train_obj['images'] = list(train_images)
        train_obj['annotations'] = list(train_annotations)

        test_obj['images'] = list(test_images)
        test_obj['annotations'] = list(test_annotations)

        data_root = ann_file.parents[0]

        with open(data_root / f'train_annotations_{fold}_{n_folds}.json', 'w') as f:
            json.dump(train_obj, f)
        with open(data_root / f'test_annotations_{fold}_{n_folds}.json', 'w') as f:
            json.dump(test_obj, f)

        train_folds.append(train_obj)
        test_folds.append(test_obj)

        fold += 1

       
    for i in range(len(train_folds)):
        for j in range(i+1, len(train_folds)):
            image_i_ids = [image['id'] for image in train_folds[i]['images']]
            image_j_ids = [image['id'] for image in train_folds[j]['images']]

            ann_i_ids = [ann['image_id'] for ann in train_folds[i]['annotations']]
            ann_j_ids = [ann['image_id'] for ann in train_folds[j]['annotations']]

            assert set(image_i_ids) != set(image_j_ids)
            assert set(ann_i_ids) != set(ann_j_ids)

    for i in range(len(test_folds)):
        for j in range(i+1, len(test_folds)):
            image_i_ids = [image['id'] for image in test_folds[i]['images']]
            image_j_ids = [image['id'] for image in test_folds[j]['images']]

            ann_i_ids = [ann['image_id'] for ann in test_folds[i]['annotations']]
            ann_j_ids = [ann['image_id'] for ann in test_folds[j]['annotations']]

            assert set(image_i_ids) != set(image_j_ids)
            assert set(ann_i_ids) != set(ann_j_ids)

def main():
    args = parse_args()

    split_coco(args.ann_file, args.fold)

if __name__ == '__main__':
    main()
