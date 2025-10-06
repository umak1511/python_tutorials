# Import libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and prepare image data
# -----------------------------

def load_images_from_folder(folder, label, image_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = imread(os.path.join(folder, filename))
            img_resized = resize(img, image_size, anti_aliasing=True)
            images.append(img_resized.flatten())  # Flatten image to 1D
            labels.append(label)
    return images, labels

# Example folder structure:
# dataset/
# ├── cats/
# ├── dogs/

data_dir = 'dataset'
categories = ['cats', 'dogs']

X, y = [], []
for label, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    imgs, lbls = load_images_from_folder(folder, label)
    X.extend(imgs)
    y.extend(lbls)

X = np.array(X)
y = np.array(y)

print(f"Dataset loaded: {len(X)} samples")

# -----------------------------
# 2. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Train Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate model
# -----------------------------
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 5. Example prediction
# -----------------------------
test_img = imread(os.path.join(data_dir, 'cats', os.listdir(os.path.join(data_dir, 'cats'))[0]))
test_img_resized = resize(test_img, (64, 64)).flatten().reshape(1, -1)
prediction = rf.predict(test_img_resized)[0]
print(f"Predicted class: {categories[prediction]}")

# -----------------------------
# 6. Visualize example
# -----------------------------
plt.imshow(test_img)
plt.title(f"Predicted: {categories[prediction]}")
plt.axis('off')
plt.show()

