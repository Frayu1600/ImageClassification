import os
import tensorflow as tf
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

# actions are already labelled
# we'll need to transform them back later on 
ACTIONS = {
    0: "do nothing",
    1: "steer left",
    2: "steer right",
    3: "gas", 
    4: "brake"
}

WIDTH = 96
HEIGHT = 96
CHANNELS = 3

# preprocessing function
def preprocess_image(image_path, augment=False):
    # load and decode the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=CHANNELS)
    img = tf.image.resize(img, (WIDTH, HEIGHT))  
    # without this it runs really badly
    img = img / 255.0  # normalize to [0, 1]

    # augmenting slightly improves results
    if augment:
        #img = tf.image.random_flip_left_right(img)  
        #img = tf.image.random_flip_up_down(img)    
        img = tf.image.random_brightness(img, max_delta=0.1, seed=seed)  
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1, seed=seed)  
        img = tf.image.random_jpeg_quality(img, min_jpeg_quality=75, max_jpeg_quality=95)  

        # add random rotation and zoom
        #img = tf.image.resize_with_crop_or_pad(img, WIDTH + 10, HEIGHT + 10)  
        #img = tf.image.random_crop(img, size=[WIDTH, HEIGHT, CHANNELS])  

    return img

def get_data(folder):
    files = []
    labels = []

    for label in range(0, len(ACTIONS)):
        for file in os.listdir(folder + f"/{label}/"):
            files.append(preprocess_image(folder + f"/{label}/{file}", augment=True))
            labels.append(label)

    return np.array(files), np.array(labels) 

# let's do a 3CL network and a 5CL one
def build_cnn_model(num_classes, num_convolutional_layers):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS))
    ])
        
    for i in range(num_convolutional_layers):
        model.add(
            tf.keras.layers.Conv2D(
                filters=2**(i+5),   # starts at 32
                kernel_size=(3, 3), 
                activation='relu', 
                padding='valid',
            ))
        model.add(tf.keras.layers.AveragePooling2D((2, 2)))
    
    # Fully connected layers
    model.add(tf.keras.layers.Flatten())

    model.add(
        tf.keras.layers.Dense(
        units=1024,   
        activation='relu'
        ))
    
    model.add(
        tf.keras.layers.Dense(
        units=1024,   
        activation='relu'
        ))
    
    model.add(tf.keras.layers.Dropout(0.8))
              
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # best for multi-class classification
    return model

def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

# read from file
argc = len(sys.argv) 
if argc < 2:
  print(f"\n[!] Usage: project.py [seed]")
  exit(1)

seed = int(sys.argv[1])

# set random seed for replication
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

train_path = "train"
test_path = "test"

# create test and train datasets
start_time = time.time()
X_train, y_train = get_data(train_path)
X_test, y_test = get_data(test_path)
end_time = time.time()
print(f"Augmentation time = {end_time-start_time}")

# TODO: fix this
#X = [X_train + X_test]
#Y = [y_train + y_test]

# split in training and validation 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)

#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=5e-4,
#    decay_steps=1000,
#    decay_rate=0.96,
#    staircase=True
#)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.4, beta_2=0.8)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
#optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=5e-2)

# Build and compile the model
num_classes = len(ACTIONS)  
model = build_cnn_model(num_classes, num_convolutional_layers=3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = []
nepochs = 15
batch_size = 256

start_time = time.time()
history.append(model.fit(X_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(X_val,y_val)))
end_time = time.time()
print(f"Training time = {end_time-start_time}")

# plot all the best models 
fig = plt.figure(figsize=(8, 6))
for h in history:
    plt.plot(h.history['accuracy'], 'r')
    plt.plot(h.history['val_accuracy'], 'b')
plt.xticks(np.arange(nepochs))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'test accuracy'], loc='upper left')
plt.suptitle(f'Model accuracy with seed {seed}', fontsize=14)
plt.show()

#score = model.evaluate(X_test, y_test)
#print("Test loss: %f" %score[0])
#print("Test accuracy: %f" %score[1])

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=list(ACTIONS.values()), digits=3))

cm = confusion_matrix(y_test, y_pred, sample_weight=None)

plt.rcParams["figure.figsize"] = (6, 6)
plot_confusion_matrix(cm, classes=list(ACTIONS.values()))

# Save the trained model
model.save("car_control_classifier.h5")