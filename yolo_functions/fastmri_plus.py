import csv
import os
import numpy as np
import torch
import torchvision
import cv2

# Function to normalize bounding box coordinates
def normalize_bbox(x, y, width, height, img_width, img_height):
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width_normalized = width / img_width
    height_normalized = height / img_height
    return x_center, y_center, width_normalized, height_normalized

def adjust_normalized_bboxes(bboxes, orig_width, orig_height, crop_width, crop_height):
    # Calculate normalized crop offsets and sizes
    x_offset = (crop_width - orig_width) / 2
    y_offset = (crop_height - orig_height) / 2

    # Adjust bounding boxes
    adjusted_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        # Adjust center coordinates
        x_new = (x + x_offset)
        y_new = (y + y_offset)
        # Scale width and height
        w_new = w
        h_new = h
        # Ensure the bounding box is within the cropped region
        # if 0 <= x_new <= 1 and 0 <= y_new <= 1:
        adjusted_bboxes.append([x_new, y_new, w_new, h_new])

    return np.array(adjusted_bboxes)


def npy_to_png(npy_data, to_pil=True, target_size=(320, 320)):
    if len(npy_data.shape) == 3:
        npy_data = np.expand_dims(npy_data, 1)
    im_abs = np.abs(npy_data)
    #im = npy_data / (np.percentile(im_abs, 99)+1e-8)
    im = np.abs(npy_data)
    im = np.expand_dims(im, 0)
    im = np.repeat(im, 3, axis=0)
    im = torch.from_numpy(im)
    im = torchvision.transforms.functional.center_crop(im, target_size)
    im = im.flip(-2)
    im = im.clamp(0, 1)
    im = im * 255
    im = im.byte()
    if to_pil:
        im = [torchvision.transforms.ToPILImage()(im)]
    return im

def load_predictions(npy_dir, image_name=None):
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    if image_name is not None:
        npy_files = [f for f in npy_files if image_name in f]
    npy_file_names = [f.split('.')[0] for f in npy_files]
    png_images = []
    image_names = []
    for f in npy_files:
        im = np.load(os.path.join(npy_dir, f))
        images = npy_to_png(im)
        png_images.extend(images)
        image_names.extend([f.split('.')[0] for i in range(len(images))])
    len_images = len(png_images)
    return png_images, image_names, len_images


def load_pngs(png_dir, image_dir, image_name=None):
    png_dir = os.path.join(png_dir, image_dir)
    png_files = [f for f in os.listdir(png_dir) if f.endswith('.png')]
    if image_name is not None:
        png_files = [f for f in png_files if image_name in f]
    #png_file_paths = [os.path.join(png_dir, f) for f in png_files]
    png_images = []
    image_names = []
    for c, f in enumerate(png_files):
        if c == 13:
            print('stop')
        im = cv2.imread(os.path.join(png_dir, f))
        png_images.append(im)
        image_names.append(f.split('.')[0])
    len_images = len(png_files)
    return png_images, image_names, len_images


def load_predictions_and_labelpreds(pred_dir, image_name=None):
    yolo_dir = os.path.join('/'.join(pred_dir.split('/')[:-1]), 'yolo_predictions_augmented')
    input_dir = os.path.join('/'.join(pred_dir.split('/')[:-1]), 'input')
    png_files, png_names, len_images = load_predictions(pred_dir, image_name=image_name)
    png_files_input, png_input_name, len_images_input = load_predictions(input_dir, image_name=image_name)
    label_paths = [os.path.join(yolo_dir, image_name + f'_{i+1:02}'+'.txt') for i in range(len_images)]  #TODO: remove +1 for other images
    return png_files, label_paths, png_files_input


def create_yolo_files(dataset='knee'):
    # File paths
    csv_file = f'./fastmri-plus/Annotations/{dataset}.csv'
    label_folder = f'./data/labels_yolo_{dataset}'  # Output folder for label files

    # Ensure the label folder exists
    os.makedirs(label_folder, exist_ok=True)

    # Step 1: Create a label map based on unique labels in the CSV
    label_map = {}

    # Step 2: Dictionary to hold data for each slice
    slice_bboxes = {}

    # Read CSV and generate label map
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            label = row['label']

            # Add new label to label_map with a unique index
            if label not in label_map:
                label_map[label] = 1
            else:
                label_map[label] += 1

    # sort label_map by value:
    label_map = {k: v for k, v in sorted(label_map.items(), key=lambda item: item[1], reverse=True)}

    # create new label_map with unique index
    label_map_index = {k: i for i, (k, v) in enumerate(label_map.items())}

    # Print the generated label map
    print("Generated Label Map:")
    for label, idx in label_map.items():
        print(f"{label}: {idx}")

    # Step 3: Now process the CSV again to convert bounding boxes to YOLO format
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            file = row['file']
            slice_number = row['slice']

            label = row['label']
            if not row['label'] == 'artifact' and not row['x'] == '':
                x = int(row['x'])
                y = int(row['y'])
                width = int(row['width'])
                height = int(row['height'])

                # Get image dimensions
                target_width, target_height = 320, 320

                # Normalize the bounding box
                x_center, y_center, width_norm, height_norm = normalize_bbox(x, y, width, height, target_width, target_height)

                # Get class index from the dynamic label_map
                class_index = label_map_index[label]

                # Generate file identifier (file + slice number)
                file_identifier = f"{file}_{int(slice_number):03d}"

                # Collect bounding boxes for the slice
                if file_identifier not in slice_bboxes:
                    slice_bboxes[file_identifier] = []

                # Append YOLO format bounding box [class_index, x_center, y_center, width_norm, height_norm]
                slice_bboxes[file_identifier].append(f"{class_index} {x_center} {y_center} {width_norm} {height_norm}")

    # Write to YOLO label files
    for file_identifier, bboxes in slice_bboxes.items():
        label_file_path = os.path.join(label_folder, f"{file_identifier}.txt")

        with open(label_file_path, 'w') as f:
            f.write('\n'.join(bboxes))

    print("Conversion to YOLO format completed.")

if __name__ == '__main__':
    create_yolo_files()
    print('annotations in yolo format created')