import torch
from model import CNN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from customDataset import EmotionsDataset
import matplotlib.pyplot as plt

# Label map for readability
label_map = {0: "Happy", 1: "Sad", 2: "Surprised"}

# Load the dataset and DataLoader
test_dataset = EmotionsDataset(r"test")
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Initialize and load the model
classifier = CNN()
classifier.load_state_dict(torch.load('cnn_emotion_weights.pth'))
classifier.eval()

all_targets = []
all_predictions = []

# Store a few images for display
images_to_display = []
preds_to_display = []
labels_to_display = []

for (data, target) in test_loader:
    prediction = classifier(data)

    targetarray = target.numpy()
    predictionarray = prediction.detach().numpy()

    print("Processing batch of 128 images...")

    for i in range(len(targetarray)):
        predictedlabel = np.argmax(predictionarray[i])
        all_targets.append(targetarray[i])
        all_predictions.append(predictedlabel)

        if len(images_to_display) < 6:
            images_to_display.append(data[i].squeeze().numpy())  # assuming grayscale
            preds_to_display.append(predictedlabel)
            labels_to_display.append(targetarray[i])

    # Stop early if we have enough for display
    if len(images_to_display) >= 6:
        break

# Display sample predictions
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images_to_display[i], cmap='gray')
    plt.title(f"True: {label_map[labels_to_display[i]]}\nPred: {label_map[preds_to_display[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Calculate and print metrics
accuracy = accuracy_score(all_targets, all_predictions)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(all_targets, all_predictions, target_names=label_map.values()))

print("Confusion Matrix:")
print(confusion_matrix(all_targets, all_predictions))
