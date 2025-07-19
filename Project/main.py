from torch.utils.data import DataLoader
from customDataset import EmotionsDataset
from PIL import Image
from model import CNN
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import torch
import matplotlib.pyplot as plt


train_dataset = EmotionsDataset(r"train")
test_dataset = EmotionsDataset(r"test")
validation_dataset = EmotionsDataset(r"validation")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=True)

classifier = CNN()
classifier.train(True)

lossfunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.0005)


epochs = 50
epoch_loss = []
validation_accuracy = []



print("\nStarting training...")


for epoch in range(epochs):

    epochloss = 0

    for (data, target) in train_loader:
        prediction = classifier(data)
        loss = lossfunction(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        epochloss += loss.item()

    epoch_loss.append(epochloss)
    print(f"Loss for epoch {epoch} is {epochloss}")



    # Validation
    print("Validation...")

    totalimages = 0
    totalcorrectpredictions = 0

    for (vdata, vtarget) in validation_loader:
        vprediction = classifier(vdata)
        vtargetarray = vtarget.numpy()
        vpredictionarray = vprediction.detach().numpy()

        for (i, label) in enumerate(vtargetarray):
            predictedlabel = np.argmax(vpredictionarray[i])
            totalimages+=1
            if (label == predictedlabel): totalcorrectpredictions+=1

        accuracy = totalcorrectpredictions / totalimages
    
    validation_accuracy.append(accuracy)
    print(f"Accuracy on validation set: {accuracy}")



# Save model weights
torch.save(classifier.state_dict(), 'cnn_emotion_weights.pth')
print("\nTraining finished")







# -------------------------- Plotting ----------------------------


# Example data
epochs = list(range(1, epochs + 1))

plt.figure()
plt.plot(epochs, validation_accuracy, marker='o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.grid(True)
plt.savefig('validation_accuracy.png')
plt.close()

# Plot and save epoch loss
plt.figure()
plt.plot(epochs, epoch_loss, marker='o', color='red', label='Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch Loss over Time')
plt.grid(True)
plt.savefig('epoch_loss.png')  # Saves the plot as a PNG file
plt.close()










# -------------------------- Testing ----------------------------

print("\nTesting...")

classifier.eval()

all_targets = []
all_predictions = []

for (data, target) in test_loader:
    prediction = classifier(data)

    targetarray = target.numpy()
    predictionarray = prediction.detach().numpy()

    print("Processing batch of 128 images...")

    for (i, label) in enumerate(targetarray):
        predictedlabel = np.argmax(predictionarray[i])
        all_targets.append(label)
        all_predictions.append(predictedlabel)



accuracy = accuracy_score(all_targets, all_predictions)
print(f"\nAccuracy: {accuracy:.4f}")

# Accuracy, precision, recall, F1
print("\nClassification Report:")
print(classification_report(all_targets, all_predictions))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(all_targets, all_predictions))

