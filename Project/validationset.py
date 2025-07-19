import os
import shutil
import random

# Paths
train_dir = 'train'
val_dir = 'validation'

# Go through each class subfolder
for class_name in os.listdir(val_dir):
    val_class_path = os.path.join(val_dir, class_name)
    train_class_path = os.path.join(train_dir, class_name)

    if not os.path.isdir(val_class_path):
        continue

    os.makedirs(train_class_path, exist_ok=True)

    # List of images in the validation subfolder
    images = [f for f in os.listdir(val_class_path) if os.path.isfile(os.path.join(val_class_path, f))]

    num_to_move = len(images) // 150
    images_to_move = random.sample(images, num_to_move)

    for img in images_to_move:
        src = os.path.join(val_class_path, img)
        dst = os.path.join(train_class_path, img)
        shutil.move(src, dst)

