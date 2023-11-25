# Import necessary libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage import io, color, transform

# Set the path to the dataset
dataset_path = "C:\\Users\\Owen\\OneDrive\\Desktop\\new folder\\train"
  # Replace with the actual path to your dataset

# Function to load and preprocess images
def load_and_preprocess_images(folder_path, num_images):
    images = []
    labels = []
    for i in range(1, num_images + 1):
        image_path = os.path.join(folder_path, f"{i}.jpg")
        img = io.imread(image_path)
        img = transform.resize(img, (50, 50), mode='constant')  # Resize images to a consistent size
        img_gray = color.rgb2gray(img)  # Convert to grayscale
        images.append(img_gray.flatten())  # Flatten the image
        labels.append(1 if "dog" in image_path else 0)  # 1 for dog, 0 for cat
    return np.array(images), np.array(labels)

# Load and preprocess the dataset
num_images = 1000  # You may adjust this based on your computational resources
X, y = load_and_preprocess_images(dataset_path, num_images)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
