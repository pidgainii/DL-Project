import os
import cv2
import random
import albumentations as A
import numpy as np

# Path to your train directory
train_dir = r'train'

augmentations = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomRotate90(p=1),
        A.Rotate(limit=45, p=1),
    ], p=1),
    A.RandomBrightnessContrast(p=1)
])


folder_image_counts = {}
for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) if f.endswith(".png")])
        folder_image_counts[folder] = count


max_count = max(folder_image_counts.values())

# Ensure all folders have the same number of images
for folder, count in folder_image_counts.items():
    if count >= max_count:
        continue

    folder_path = os.path.join(train_dir, folder)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    next_img_index = count

    while len(image_files) < max_count:
        img_name = random.choice(image_files)
        img_path = os.path.join(folder_path, img_name)

        # Read as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Albumentations expects HxWxC, so expand dims
        image_input = np.expand_dims(image, axis=2)

        # Apply augmentations
        augmented = augmentations(image=image_input)['image']

        # Squeeze back to HxW (remove channel dim)
        augmented_gray = np.squeeze(augmented)

        # Ensure it's still uint8
        if augmented_gray.dtype != np.uint8:
            augmented_gray = np.clip(augmented_gray, 0, 255).astype(np.uint8)

        new_img_name = f"im{next_img_index}.png"
        new_img_path = os.path.join(folder_path, new_img_name)
        cv2.imwrite(new_img_path, augmented_gray)

        image_files.append(new_img_name)
        next_img_index += 1

    print("Processing...")

print("All folders are now balanced with grayscale images.")
