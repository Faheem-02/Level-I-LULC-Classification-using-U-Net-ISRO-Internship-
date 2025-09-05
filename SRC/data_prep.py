!pip install split-folders

# -----------

!pip install patchify

# -----------

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import splitfolders
import random

# -----------

# Quick understanding of the dataset
temp_img = cv2.imread("/kaggle/input/landcoverai/images/M-34-51-C-d-4-1.tif")
plt.imshow(temp_img[:, :, 1])
temp_mask = cv2.imread("/kaggle/input/landcoverai/masks/M-34-51-C-d-4-1.tif")
labels, count = np.unique(temp_mask[:, :, 0], return_counts=True)
print("Labels are: ", labels, " and the counts are: ", count)

# -----------

# Crop each large image into patches
root_directory = '/kaggle/working/'
patch_size = 256
img_dir = "/kaggle/input/landcoverai/images/"
os.makedirs(root_directory + "256_patches/images/", exist_ok=True)
for path, subdirs, files in os.walk(img_dir):
    dirname = path.split(os.path.sep)[-1]
    images = os.listdir(path)
    for i, image_name in enumerate(images):
        if image_name.endswith(".tif"):
            image = cv2.imread(path + "/" + image_name, 1)
            if image is None:
                print(f"Failed to load image: {path}/{image_name}")
                continue
            SIZE_X = (image.shape[1] // patch_size) * patch_size
            SIZE_Y = (image.shape[0] // patch_size) * patch_size
            image = Image.fromarray(image)
            image = image.crop((0, 0, SIZE_X, SIZE_Y))
            image = np.array(image)
            print("Now patchifying image:", path + "/" + image_name)
            patches_img = patchify(image, (256, 256, 3), step=256)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :]
                    single_patch_img = single_patch_img[0]
                    cv2.imwrite(root_directory + "256_patches/images/" +
                                image_name + "patch_" + str(i) + str(j) + ".tif", single_patch_img)

# Crop each large mask into patches
mask_dir = "/kaggle/input/landcoverai/masks/"
os.makedirs(root_directory + "256_patches/masks/", exist_ok=True)
for path, subdirs, files in os.walk(mask_dir):
    dirname = path.split(os.path.sep)[-1]
    masks = os.listdir(path)
    for i, mask_name in enumerate(masks):
        if mask_name.endswith(".tif"):
            mask = cv2.imread(path + "/" + mask_name, 0)
            if mask is None:
                print(f"Failed to load mask: {path}/{mask_name}")
                continue
            SIZE_X = (mask.shape[1] // patch_size) * patch_size
            SIZE_Y = (mask.shape[0] // patch_size) * patch_size
            mask = Image.fromarray(mask)
            mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
            mask = np.array(mask)
            # Clip mask values to ensure they are within [0, 3]
            mask = np.clip(mask, 0, 3)
            print("Now patchifying mask:", path + "/" + mask_name)
            patches_mask = patchify(mask, (256, 256), step=256)
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i, j, :, :]
                    cv2.imwrite(root_directory + "256_patches/masks/" +
                                mask_name + "patch_" + str(i) + str(j) + ".tif", single_patch_mask)

train_img_dir = root_directory + "256_patches/images/"
train_mask_dir = root_directory + "256_patches/masks/"

# -----------

# Verify a sample patched image and mask
img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)
img_num = random.randint(0, len(img_list) - 1)
img_for_plot = cv2.imread(train_img_dir + img_list[img_num], 1)
img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)
mask_for_plot = cv2.imread(train_mask_dir + msk_list[img_num], 0)
plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Sample Patched Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Sample Patched Mask')
plt.show()
print("Unique mask values:", np.unique(mask_for_plot))

# -----------

# Copy useful images and masks
os.makedirs(root_directory + '256_patches/images_with_useful_info/images/', exist_ok=True)
os.makedirs(root_directory + '256_patches/images_with_useful_info/masks/', exist_ok=True)
useless = 0
for img in range(len(img_list)):
    img_name = img_list[img]
    mask_name = msk_list[img]
    print("Now preparing image and masks number: ", img)
    temp_image = cv2.imread(train_img_dir + img_list[img], 1)
    temp_mask = cv2.imread(train_mask_dir + msk_list[img], 0)
    # Ensure mask values are within [0, 3]
    temp_mask = np.clip(temp_mask, 0, 3)
    val, counts = np.unique(temp_mask, return_counts=True)
    if (1 - (counts[0] / counts.sum())) > 0.05:
        print("Save Me")
        cv2.imwrite(root_directory + '256_patches/images_with_useful_info/images/' + img_name, temp_image)
        cv2.imwrite(root_directory + '256_patches/images_with_useful_info/masks/' + mask_name, temp_mask)
    else:
        print("I am useless")
        useless += 1

print("Total useful images are: ", len(img_list) - useless)
print("Total useless images are: ", useless)

# -----------

# Split into train, validation, and test
os.makedirs(root_directory + 'data_for_training_and_testing/', exist_ok=True)
input_folder = root_directory + '256_patches/images_with_useful_info/'
output_folder = root_directory + 'data_for_training_and_testing/'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.2, 0.1), group_prefix=None)

# -----------

# Verify the split data
train_img_dir_final = output_folder + "train/images/"
train_mask_dir_final = output_folder + "train/masks/"
val_img_dir_final = output_folder + "val/images/"
val_mask_dir_final = output_folder + "val/masks/"
test_img_dir_final = output_folder + "test/images/"
test_mask_dir_final = output_folder + "test/masks/"

img_list_train = os.listdir(train_img_dir_final)
msk_list_train = os.listdir(train_mask_dir_final)
img_list_val = os.listdir(val_img_dir_final)
msk_list_val = os.listdir(val_mask_dir_final)
img_list_test = os.listdir(test_img_dir_final)
msk_list_test = os.listdir(test_mask_dir_final)

img_num_train = random.randint(0, len(img_list_train) - 1)
img_num_val = random.randint(0, len(img_list_val) - 1)
img_num_test = random.randint(0, len(img_list_test) - 1)

img_for_plot_train = cv2.imread(train_img_dir_final + img_list_train[img_num_train], 1)
img_for_plot_train = cv2.cvtColor(img_for_plot_train, cv2.COLOR_BGR2RGB)
mask_for_plot_train = cv2.imread(train_mask_dir_final + msk_list_train[img_num_train], 0)

img_for_plot_val = cv2.imread(val_img_dir_final + img_list_val[img_num_val], 1)
img_for_plot_val = cv2.cvtColor(img_for_plot_val, cv2.COLOR_BGR2RGB)
mask_for_plot_val = cv2.imread(val_mask_dir_final + msk_list_val[img_num_val], 0)

img_for_plot_test = cv2.imread(test_img_dir_final + img_list_test[img_num_test], 1)
img_for_plot_test = cv2.cvtColor(img_for_plot_test, cv2.COLOR_BGR2RGB)
mask_for_plot_test = cv2.imread(test_mask_dir_final + msk_list_test[img_num_test], 0)

plt.figure(figsize=(12, 12))
plt.subplot(331)
plt.imshow(img_for_plot_train)
plt.title('Train Image')
plt.subplot(332)
plt.imshow(mask_for_plot_train, cmap='gray')
plt.title('Train Mask')
plt.subplot(333)
plt.imshow(img_for_plot_val)
plt.title('Val Image')
plt.subplot(334)
plt.imshow(mask_for_plot_val, cmap='gray')
plt.title('Val Mask')
plt.subplot(335)
plt.imshow(img_for_plot_test)
plt.title('Test Image')
plt.subplot(336)
plt.imshow(mask_for_plot_test, cmap='gray')
plt.title('Test Mask')
plt.show()
print("Unique train mask values:", np.unique(mask_for_plot_train))
print("Unique val mask values:", np.unique(mask_for_plot_val))
print("Unique test mask values:", np.unique(mask_for_plot_test))

# -----------

train_mask_dir = "/kaggle/working/data_for_training_and_testing/train/masks"
val_mask_dir = "/kaggle/working/data_for_training_and_testing/val/masks"

# Check train masks
train_masks = os.listdir(train_mask_dir)
for mask_name in train_masks[:5]:  # Check a few samples
    mask = cv2.imread(os.path.join(train_mask_dir, mask_name), 0)
    print(f"Train mask {mask_name} unique values:", np.unique(mask))

# Check val masks
val_masks = os.listdir(val_mask_dir)
for mask_name in val_masks[:5]:  # Check a few samples
    mask = cv2.imread(os.path.join(val_mask_dir, mask_name), 0)
    print(f"Val mask {mask_name} unique values:", np.unique(mask))

# -----------

