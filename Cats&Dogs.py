# %% [markdown]
# let's start by importing libraries first:

# %%
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# loading the data and preparing the dataset

# %%
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')

#defining image size
Img_size = 64 #size which images will be resized

#creating a fucntion to load images
def load_images(data):
    images=[]
    labels=[]
    for index, row in data.iterrows():
        img_path = row['image:FILE'] #path to the image
        img = cv2.imread(img_path)
        img = cv2.resize(img,(Img_size,Img_size))
        images.append(img)
        labels.append(row['category'])
    return np.array(images), np.array(labels)

#load training and validation images
x_train, y_train = load_images(train_data)
x_val,y_val = load_images(val_data)

# %% [markdown]
# preprocessing the data

# %%
#flattenign the images
x_train = x_train.reshape(x_train.shape[0],-1)
#Reshape to (num_samples, height*width*channels)
x_val = x_val.reshape(x_val.shape[0], -1)

# %% [markdown]
# trainig the svm model

# %%
model = SVC(kernel='linear')
model.fit(x_train,y_train)

#predictin the values
y_pred = model.predict(x_val)

print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val, y_pred))

# %% [markdown]
# This represents the following:
# 
# True Positives (TP): 4 (Correctly predicted positives)
# False Negatives (FN): 20 (Positives incorrectly predicted as negatives)
# False Positives (FP): 15 (Negatives incorrectly predicted as positives)
# True Negatives (TN): 31 (Correctly predicted negatives)

# %% [markdown]
# for visualizing part 
# Visualize the confusion matrix and display sample images with their predictions.

# %%
cm = confusion_matrix(y_val, y_pred)

#plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# %%
def plot_images(images, titles, cols=3):
    """Display images in a grid with titles."""
    rows = len(images) // cols + (len(images) % cols > 0)
    plt.figure(figsize=(15, 5 * rows))
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Select a few images to display
sample_indices = np.random.choice(len(x_val), size=9, replace=False)
sample_images = [x_val[i].reshape(Img_size, Img_size, 3) for i in sample_indices]
sample_titles = [f'Actual: {"Dog" if y_val[i] == 1 else "Cat"}\nPredicted: {"Dog" if y_pred[i] == 1 else "Cat"}' for i in sample_indices]

# Plot the images
plot_images(sample_images, sample_titles)



