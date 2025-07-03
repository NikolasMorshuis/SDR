import os
import numpy as np
import yaml
import ultralytics.utils.ops as ops
import pandas as pd

label_map = 'data/label_map.yaml'
# import label_map:
with open(label_map, 'r') as f:
    label_map_dict = yaml.load(f, Loader=yaml.FullLoader)

label_map_short = 'data/label_map_short.yaml'
with open(label_map_short, 'r') as f:
    label_map_dict_short = yaml.load(f, Loader=yaml.FullLoader)

def increase_boxsize_xywh_fn(box, increase_boxsize=10, imgsz=320):
    # Add a margin to the box
    x, y, w, h = box
    addition = increase_boxsize/imgsz
    w = w + addition
    h = h + addition
    return [x, y, w, h]

def get_reconstructed_element_names(dir, endswith='.npy'):
    # reads the detected elements from the given directory
    # list .npy files in dir:
    files = [f for f in os.listdir(dir) if f.endswith(endswith)]
    files = [f.split('.')[0] for f in files]
    file_names_root = ['_'.join(f.split('_')[:2]) for f in files]
    file_names_root = np.unique(file_names_root)
    return file_names_root, files

def load_prediction_boxes(dir_root, label_dir, image_name, imgsz=320, increase_boxsize=0, confidence_threshold=0.15):
    pred_file = os.path.join(dir_root, label_dir, image_name+'.txt')
    if os.path.exists(pred_file):
        with open(pred_file, 'r') as f:
            lines = f.readlines()
        boxes = []
        for line in lines:
            box = line.split(' ')
            box = [float(b) for b in box]
            boxes.append(box)
        # change boxes to xyxy format:
        classes = [int(box[0]) for box in boxes]
        if len(box) > 5:
            confidence = [box[5] for box in boxes]
        else:
            confidence = [None for box in boxes]
        class_names = [label_map_dict[c] for c in classes]
        if increase_boxsize and increase_boxsize > 0:
            boxes = [increase_boxsize_xywh_fn(box[1:5], increase_boxsize, imgsz) for box in boxes]
        else:
            boxes = [box[1:] for box in boxes]
        boxes = [ops.xywh2xyxy(np.array(box[0:4])*imgsz) for box in boxes]
        boxes = np.array(boxes)
        if confidence_threshold and len(box) > 5:
            indices = [i for i, c in enumerate(confidence) if c > confidence_threshold]
            boxes = boxes[indices]
            classes = [classes[i] for i in indices]
            confidence = [confidence[i] for i in indices]
            class_names = [class_names[i] for i in indices]
    else:
        boxes = []
        classes = []
        confidence = []
        class_names = []
    return boxes, classes, confidence, class_names

def load_gt_boxes(dir_root, label_dir, image_name, imgsz=320, increase_boxsize=0):
    #TODO: combine with load_prediction_boxes
    with open(os.path.join(label_path, '_'.join(image_name.split('_')[:2])+'.txt'), 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        box = line.split(' ')
        box = [float(b) for b in box]
        boxes.append(box)
    # change boxes to xyxy format:
    classes = [int(box[0]) for box in boxes]
    if increase_boxsize:
        boxes = [increase_boxsize_xywh_fn(box[1:], increase_boxsize, imgsz) for box in boxes]
    else:
        boxes = [box[1:] for box in boxes]
    boxes = [ops.xywh2xyxy(np.array(box)*imgsz) for box in boxes]
    boxes = np.array(boxes)
    # confidence 1 for all boxes:
    confidence = [None for box in boxes]
    class_names = [label_map_dict[c] for c in classes]
    return boxes, classes, confidence, class_names


def select_best_recons(recon_root_dir, element_name, metric='psnr'):
    # get all recon dirs:
    best_recons = []
    recon_metrics = pd.read_csv(os.path.join(recon_root_dir, recon_root_dir, 'recon_metrics.csv'))
    recon_metrics = recon_metrics[recon_metrics['element'].str.contains(element_name)]
    values = recon_metrics[metric].values
    indices = np.argsort(values)[::-1]
    recon_metrics = recon_metrics.sort_values(by=metric, ascending=False)
    return indices, recon_metrics, values[indices]


if __name__ == '__main__':
    pass
