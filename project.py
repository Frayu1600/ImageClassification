import os
import tensorflow as tf
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

augment=False

# preprocessing function
def preprocess_image(image_path, augment):
    # load and decode the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=CHANNELS)
    img = tf.image.resize(img, (WIDTH, HEIGHT))  
    # without this it runs really badly
    img = img / 255.0  # normalize to [0, 1]from sklearn.utils.class_weight import compute_class_weight

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
            files.append(preprocess_image(folder + f"/{label}/{file}", augment=augment))
            labels.append(label)

    return np.array(files), np.array(labels) 

# let's do a 3CL network and a 5CL one
def build_cnn_model(num_classes, num_convolutional_layers, num_dense_layers, dense_units, dropout, activation, padding):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS))
    ])
        
    for i in range(num_convolutional_layers):
        model.add(
            tf.keras.layers.Conv2D(
                filters=2**(i+5),   # starts at 32
                kernel_size=(2, 2), 
                activation=activation,
                padding=padding,    # valid
            ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    # Fully connected layers
    model.add(tf.keras.layers.Flatten())

    for i in range(num_dense_layers):
        model.add(tf.keras.layers.Dense(
            units=dense_units[i],
        activation=activation))
    
        if i != num_dense_layers:
            model.add(tf.keras.layers.Dropout(dropout))
           
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

def hyperparameter_search(num_convolutional_layers, activation, padding, optimizer):

  tuner = kt.RandomSearch(
   lambda hp: build_cnn_model(num_convolutional_layers, activation, padding, hp, optimizer),
   objective='accuracy',      
   max_trials=4,             # hyperparameter combinations to try
   executions_per_trial=2,   # models built per trial
   directory='hyperparameter_search_' + optimizer.lower(),
  )

  tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=nepochs, batch_size=batch_size)
  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
  
  print(f"Best hyperparameters for {optimizer}: \n {best_hps.values}")

  return tuner.hypermodel.build(best_hps), best_hps.values

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

adam = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.6, beta_2=0.8)
rmsprop = tf.keras.optimizers.RMSprop(learning_rate=1e-4, momentum=0.8, rho=0.8)
sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

# Build and compile the model
num_classes = len(ACTIONS)  
model_adam = build_cnn_model(num_classes, num_convolutional_layers=3, num_dense_layers=2, 
                             dense_units=[256, 64], dropout=0.5, activation='relu', padding='valid')
model_adam.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_adam.summary()

model_rmsprop = build_cnn_model(num_classes, num_convolutional_layers=3, num_dense_layers=2, 
                                dense_units=[256, 64], dropout=0.5, activation='relu', padding='valid')
model_rmsprop.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_rmsprop.summary()

model_sgd = build_cnn_model(num_classes, num_convolutional_layers=3, num_dense_layers=2, 
                            dense_units=[256, 64], dropout=0.5, activation='relu', padding='valid')
model_sgd.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_sgd.summary()

histories = []
nepochs = 10
batch_size = 64

start_time_adam = time.time()
histories.append(model_adam.fit(X_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(X_val,y_val)))#, class_weight=class_weight_dict))
end_time_adam = time.time()

start_time_rmsprop = time.time()
histories.append(model_rmsprop.fit(X_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(X_val,y_val)))#, class_weight=class_weight_dict))
end_time_rmsprop = time.time()

start_time_sgd = time.time()
histories.append(model_sgd.fit(X_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(X_val,y_val)))#, class_weight=class_weight_dict))
end_time_sgd = time.time()

titles = ["Adam", "RMSProp", "SGD"]

print(f"Training time Adam = {end_time_adam-start_time_adam}")
print(f"Training time RMSProp = {end_time_rmsprop-start_time_rmsprop}")
print(f"Training time SGD = {end_time_sgd-start_time_sgd}")


augment_label = "with augmentation"
if augment == False:
    augment_label = "without augmentation"


# plot all the best models 
fig = plt.figure(figsize=(20, 6))
for i, h in enumerate(histories):
 plt.subplot(1,3,i+1)
 plt.plot(h.history['accuracy'], 'r')
 plt.plot(h.history['val_accuracy'], 'b')
 plt.title(titles[i])
 plt.ylim(0.25, 0.65)
 plt.ylabel('accuracy')
 plt.xlabel('epoch')
 plt.legend(['train mse', 'test mse'], loc='upper left')
 plt.suptitle(f'Model 1 accuracy with seed {seed} ({augment_label})', fontsize=14)
plt.show()

y_pred_adam = np.argmax(model_adam.predict(X_test), axis=1)
y_pred_rmsprop = np.argmax(model_rmsprop.predict(X_test), axis=1)
y_pred_sgd = np.argmax(model_sgd.predict(X_test), axis=1)

print(classification_report(y_test, y_pred_adam, target_names=list(ACTIONS.values()), digits=3))
print(classification_report(y_test, y_pred_rmsprop, target_names=list(ACTIONS.values()), digits=3))
print(classification_report(y_test, y_pred_sgd, target_names=list(ACTIONS.values()), digits=3))

cm_adam = confusion_matrix(y_test, y_pred_adam)
cm_rmsprop = confusion_matrix(y_test, y_pred_rmsprop)
cm_sgd = confusion_matrix(y_test, y_pred_sgd)

cm_adam_display = ConfusionMatrixDisplay(confusion_matrix=cm_adam, display_labels=list(ACTIONS.values()))
cm_rmsprop_display = ConfusionMatrixDisplay(confusion_matrix=cm_rmsprop, display_labels=list(ACTIONS.values()))
cm_sgd_display = ConfusionMatrixDisplay(confusion_matrix=cm_sgd, display_labels=list(ACTIONS.values()))

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
cm_adam_display.plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title("Adam")

cm_rmsprop_display.plot(ax=axes[1], cmap=plt.cm.Blues)
axes[1].set_title("RMSProp")

cm_sgd_display.plot(ax=axes[2], cmap=plt.cm.Blues)
axes[2].set_title("SGD")

plt.suptitle(f'Model 1 confusion matrixes with seed {seed} ({augment_label})', fontsize=14)
plt.tight_layout()
plt.show()

# Save the trained model
model_adam.save("car_control_classifier_adam.h5")
model_rmsprop.save("car_control_classifier_rmsprop.h5")
model_sgd.save("car_control_classifier_sgd.h5")