import os
import sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
import seaborn as sns
import pandas as pd
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

# preprocessing function
def preprocess_image(image_path, augment):
    # load and decode the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=CHANNELS)
    img = tf.image.resize(img, (WIDTH, HEIGHT))  
    img = img / 255.0  # normalize to [0, 1]

    if augment: 
        img = tf.image.random_brightness(img, max_delta=0.1, seed=seed)  
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1, seed=seed)  
        img = tf.image.random_jpeg_quality(img, min_jpeg_quality=75, max_jpeg_quality=95)  

    return img

def get_data(folder):
    files = []
    labels = []

    for label in range(0, len(ACTIONS)):
        for file in os.listdir(folder + f"/{label}/"):
            files.append(preprocess_image(folder + f"/{label}/{file}", augment=(augment == "aug")))
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
                filters=2**(i+5),       # era i+5
                kernel_size=(2, 2),     # era 2,2
                activation=activation,
                padding=padding,        # era valid
            ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))     # era 2,2
    
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

def build_optimized_model(hp, optimizer_name, build_model):
    if optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 5e-5, 1e-4),
            beta_1=hp.Float("beta_1", 0.3, 0.8),
            beta_2=hp.Float("beta_2", 0.4, 0.9),
        )
    elif optimizer_name == "RMSProp":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.Float('learning_rate', 5e-5, 1e-4),
            momentum=hp.Float('momentum', 0.5, 0.9),
            rho=hp.Float('rho', 0.6, 0.9),
        )
    elif optimizer_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=hp.Float('learning_rate', 5e-4, 5e-2),
            momentum=hp.Float('momentum', 0.5, 0.9),
        )
    else:
        print(f"[!] Optimizer '{optimizer_name}' not found, please specify either Adam, RMSProp or SGD.")
        exit(1)

    model = build_model()
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def hyperparameter_search(optimizer_name, build_model):

    tuner = kt.Hyperband(
        lambda hp: build_optimized_model(hp, optimizer_name, build_model),
        objective="val_accuracy",
        max_epochs=nepochs,
        factor=3,
        directory=f"hps/{model}/hyperparameter_search_{optimizer_name.lower()}",
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=nepochs, batch_size=batch_size)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Best hyperparameters for optimizer {optimizer_name}: {best_hps.values}")

    return tuner.hypermodel.build(best_hps), best_hps.values

def build_model1():
   return build_cnn_model(num_classes, num_convolutional_layers=3, num_dense_layers=2, 
                             dense_units=[256, 64], dropout=0.5, activation='relu', padding='valid')

def build_model2():
   return build_cnn_model(num_classes, num_convolutional_layers=4, num_dense_layers=1, 
                                        dense_units=[512], dropout=0.7, activation='tanh', padding='same')
   
def plot_classification_report():
    report_adam = classification_report(y_test, y_pred_adam, target_names=list(ACTIONS.values()), digits=3, output_dict=True)
    report_rmsprop = classification_report(y_test, y_pred_rmsprop, target_names=list(ACTIONS.values()), digits=3, output_dict=True)
    report_sgd = classification_report(y_test, y_pred_sgd, target_names=list(ACTIONS.values()), digits=3, output_dict=True)

    report_df_adam = pd.DataFrame.from_dict(report_adam).transpose()
    report_df_adam = report_df_adam.drop(index=["accuracy", "macro avg", "weighted avg"], columns=["support"], errors="ignore")

    report_df_rmsprop = pd.DataFrame.from_dict(report_rmsprop).transpose()
    report_df_rmsprop = report_df_rmsprop.drop(index=["accuracy", "macro avg", "weighted avg"], columns=["support"], errors="ignore")

    report_df_sgd = pd.DataFrame.from_dict(report_sgd).transpose()
    report_df_sgd = report_df_sgd.drop(index=["accuracy", "macro avg", "weighted avg"], columns=["support"], errors="ignore")

    # Plot the heatmap
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sns.heatmap(report_df_adam, annot=True, cmap="Greens", fmt=".2f", vmin=0.0, vmax=1.0, cbar=True, ax=axes[0])
    axes[0].set_title("Adam")

    sns.heatmap(report_df_rmsprop, annot=True, cmap="Greens", fmt=".2f", vmin=0.0, vmax=1.0, cbar=True, ax=axes[1])
    axes[1].set_title("RMSProp")

    sns.heatmap(report_df_sgd, annot=True, cmap="Greens", fmt=".2f", vmin=0.0, vmax=1.0, cbar=True, ax=axes[2])
    axes[2].set_title("SGD")

    plt.suptitle(f'{model_label} Classification report heatmap ({augment_label})', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrixes():
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

    plt.suptitle(f'{model_label} confusion matrixes with seed {seed} ({augment_label})', fontsize=14)
    plt.tight_layout()
    plt.show()

# read from file
argc = len(sys.argv) 
if argc < 4:
  print(f"\n[!] Usage: project.py [seed] [model1/model2] [aug/noaug]")
  exit(1)

seed = int(sys.argv[1].strip())
model = sys.argv[2].strip().lower()
augment = sys.argv[3].strip().lower()

if model not in ["model1", "model2"]:
   print("[!] Please select either 'model1' or 'model2'")

if augment not in ["aug", "noaug"]:
   print("[!] Please select either 'aug' or 'noaug'")

augment_label = "with augmentation"
aug_file_label = augment
if augment == "noaug":
    augment_label = "without augmentation"

model_label = "Model 1"
if model == "model2":
   model_label = "Model 2"

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

# split in training and validation 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)

# compile models
num_classes = len(ACTIONS) 
nepochs = 10 
batch_size = 64

if model == "model1":
    adam = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.6, beta_2=0.8)
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=1e-4, momentum=0.8, rho=0.8)
    sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

    model_adam = build_model1()
    model_rmsprop = build_model1()
    model_sgd = build_model1()

    model_adam.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_rmsprop.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_sgd.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #model_adam, best_parameters_adam = hyperparameter_search("Adam", build_model1)
    #model_rmsprop, best_parameters_rmsprop = hyperparameter_search("RMSProp", build_model1)
    #model_sgd, best_parameters_sgd = hyperparameter_search("SGD", build_model1)
else:
    #model_adam = build_model2()
    #model_rmsprop = build_model2()
    #model_sgd = build_model2()

    model_adam, best_parameters_adam = hyperparameter_search("Adam", build_model2)
    model_rmsprop, best_parameters_rmsprop = hyperparameter_search("RMSProp", build_model2)
    model_sgd, best_parameters_sgd = hyperparameter_search("SGD", build_model2)

# they're all the same anyway 
model_sgd.summary()

# training time!
histories = []

start_time_adam = time.time()
histories.append(model_adam.fit(X_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(X_val,y_val)))#, class_weight=class_weight_dict))
end_time_adam = time.time()

start_time_rmsprop = time.time()
histories.append(model_rmsprop.fit(X_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(X_val,y_val)))#, class_weight=class_weight_dict))
end_time_rmsprop = time.time()

start_time_sgd = time.time()
histories.append(model_sgd.fit(X_train, y_train, batch_size=batch_size, epochs=nepochs, verbose=1, validation_data=(X_val,y_val)))#, class_weight=class_weight_dict))
end_time_sgd = time.time()

print(f"Training time Adam = {end_time_adam-start_time_adam}")
print(f"Training time RMSProp = {end_time_rmsprop-start_time_rmsprop}")
print(f"Training time SGD = {end_time_sgd-start_time_sgd}")

# prepare labels 
titles = ["Adam", "RMSProp", "SGD"]

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
 plt.suptitle(f'{model_label} accuracy with seed {seed} ({augment_label})', fontsize=14)
plt.show()

# print reports for each model
y_pred_adam = np.argmax(model_adam.predict(X_test), axis=1)
y_pred_rmsprop = np.argmax(model_rmsprop.predict(X_test), axis=1)
y_pred_sgd = np.argmax(model_sgd.predict(X_test), axis=1)

plot_classification_report()

plot_confusion_matrixes()

# save each trained model
model_adam.save(f"car_control_classifier_adam_{aug_file_label}.h5")
model_rmsprop.save(f"car_control_classifier_rmsprop_{aug_file_label}.h5")
model_sgd.save(f"car_control_classifier_sgd_{aug_file_label}.h5")