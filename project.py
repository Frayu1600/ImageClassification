import os
import tensorflow as tf
import numpy as np
import sys
import time
from sklearn.model_selection import train_test_split

# actions are already labelled
# we'll need to transform them back later on 
ACTIONS = {
    0: "do nothing",
    1: "steer left",
    2: "steer right",
    3: "gas", 
    4: "brake"
}

# preprocessing function
def preprocess_image(image_path):
    # load and decode the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (96, 96))  
    img = img / 255.0  # normalize to [0, 1]
    return img

# builds a dataset in the form of 
# filename;x
def get_data(folder):
    files = []
    labels = []

    for label in range(0, len(ACTIONS)):
        for file in os.listdir(folder + f"/{label}/"):
            files.append(preprocess_image(folder + f"/{label}/{file}"))
            labels.append(label)

    return files, labels 

def build_cnn_model(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(96, 96, 3)),
        
        # Convolutional layers
        tf.keras.layers.Conv2D(
            32, 
            (3, 3), 
            activation='relu', 
            padding='same'
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, 
            (3, 3), 
            activation='relu', 
            padding='same'
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, 
            (3, 3), 
            activation='relu', 
            padding='same'
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Fully connected layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Output layer: Predict action class
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Softmax for classification
    ])
    return model

# read from file
argc = len(sys.argv) 
if argc < 2:
  print(f"\n[!] Usage: project.py [seed]")
  exit(1)

seed = int(sys.argv[1])
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

train_path = "train"
test_path = "test"

# create test and train datasets
X_train, y_train = get_data(train_path)
X_test, y_test = get_data(test_path)

X = [X_train + X_test]
Y = [y_train + y_test]

# split in training and validation 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)

# Build and compile the model
num_classes = len(ACTIONS)  # Number of unique actions
model = build_cnn_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

histories = []
nepochs = 100
batch_size = 200

# validation set?
start_time = time.time()
#histories.append(model.fit(X_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(X_val,y_val)))
end_time = time.time()
print(f"Training time = {end_time-start_time}")

# Save the trained model
#model.save("car_control_classifier.h5")