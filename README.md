# Deeo Learning Project

This project is a python implementation (using Pytorch) of a model used to recognize emotions in human faces

## Dataset Description

In this classification problem, we will use a dataset with images of faces, with 3 different emotions: happy, sad and surprised. The original dataset contains a folder for training and another one for testing. There are more images in the happy class, therefore data augmentation will be performed on the other two classes in order to have a balanced dataset. Augmentation will only be performed on the training set. Testing set will remain as it was (unequal number of images per class). Also, images will be extracted from the training set, to create the validation set. This is to be used throughout the training process to assess the model’s performance. The validation set size will be small, as we don’t need a big set of images to verify the model’s performance. The validation and test set images will not be used to train the model

The processed dataset contains:
  - Training: 7169 images per class (21507 images in total)
  - Validation: 46 images per class (138 images in total)
  - Testing: 1774 images for happy class, 1247 images for sad class, 831 images for surprised class.
The images are in grayscale format (only one channel). They have size 48*48 pixels
