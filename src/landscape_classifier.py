##### LOAD MODULES
# basic tools
import os
import pandas as pd
import numpy as np

# image preprocessing
import cv2

# tensorflow tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# early stopping
from tensorflow.keras.callbacks import EarlyStopping

#scikit-learn
from sklearn.metrics import classification_report

# for plotting
import matplotlib.pyplot as plt

##### PLOTTING FUNCTION
# courtesy of Ross
# This function was provided as part of the Visual Analytics course
# in the Cultural Data Science elective at Aarhus University.
def plot_history(H, epochs):

    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.plot(
        np.arange(0, epochs), 
        H.history["loss"], 
        label = "train_loss")
    plt.plot(
        np.arange(0, epochs), 
        H.history["val_loss"], 
        label = "val_loss", 
        linestyle = ":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(
        np.arange(0, epochs), 
        H.history["accuracy"], 
        label = "train_acc")
    plt.plot(
        np.arange(0, epochs), 
        H.history["val_accuracy"], 
        label = "val_acc", 
        linestyle = ":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    plt.savefig(os.path.join(
        "out", 
        "plot.jpg"))

##### SAVE MODEL SUMMARY FUNCTION
def save_summary(s):
    with open(os.path.join("out", "model_summary.txt"), 'a') as f:
        print(s, file = f)

##### MAIN
def main():

    # CONFIGURE DIRECTORIES

    # Overall location of data.
    base_dir = os.path.join(
        "..",
        "data")

    # Location of train, test and validation folders.
    train_dir = os.path.join(
        base_dir,
        "train")
    val_dir = os.path.join(
        base_dir,
        "val")
    test_dir = os.path.join(
        base_dir,
        "test")

    # Location of categories in train split.
    train_coast_dir = os.path.join(
        train_dir,
        "Coast")
    train_desert_dir = os.path.join(
        train_dir,
        "Desert")
    train_forest_dir = os.path.join(
        train_dir,
        "Forest")
    train_glacier_dir = os.path.join(
        train_dir,
        "Glacier")
    train_mountain_dir = os.path.join(
        train_dir,
        "Mountain")

    # Location of categories in validation split.
    val_coast_dir = os.path.join(
        val_dir,
        "Coast")
    val_desert_dir = os.path.join(
        val_dir,
        "Desert")
    val_forest_dir = os.path.join(
        val_dir,
        "Forest")
    val_glacier_dir = os.path.join(
        val_dir,
        "Glacier")
    val_mountain_dir = os.path.join(
        val_dir,
        "Mountain")

    # Location of categories within test split.
    test_coast_dir = os.path.join(
        test_dir,
        "Coast")
    test_desert_dir = os.path.join(
        test_dir,
        "Desert")
    test_forest_dir = os.path.join(
        test_dir,
        "Forest")
    test_glacier_dir = os.path.join(
        test_dir,
        "Glacier")
    test_mountain_dir = os.path.join(
        test_dir,
        "Mountain")

    
    # LOAD MODEL
    # load model without classifier layers
    model = VGG16(
        include_top = False, 
        pooling = 'avg',
        input_shape = (224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(
        256, 
        activation = 'relu')(bn)
    class2 = Dense(
        128, 
        activation = 'relu')(class1)
    output = Dense(
        5, 
        activation = 'softmax')(class2)

    # define new model
    model = Model(
        inputs = model.inputs, 
        outputs = output)

    # configure learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.01,
        decay_steps = 10000,
        decay_rate = 0.9)
    sgd = SGD(learning_rate = lr_schedule)

    # compile model
    model.compile(
        optimizer = sgd,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

    # summarize
    model.summary(print_fn = save_summary)

    print(model.summary())

    # IMAGE PREPROCESSING + DATA AUGMENTATION

    # Some necessary variables
    IMG_SHAPE = 224
    batch_size = 128

    # Define data generator. 
    # Normalize image values.
    # Allow for horizontal flipping and rotation (data augmentation).
    datagen = ImageDataGenerator(
        rescale = 1./255, 
        horizontal_flip = True, 
        rotation_range = 20,
        preprocessing_function = lambda x: tf.image.resize(
            x, 
            (IMG_SHAPE, IMG_SHAPE)))

    # Training - flow from folder. 
    train_data_gen = datagen.flow_from_directory(
        directory = train_dir,
        batch_size = batch_size,
        target_size = (IMG_SHAPE, IMG_SHAPE),
        shuffle = True,
        class_mode = 'categorical')

    # Validation - flow from folder.
    val_data_gen = datagen.flow_from_directory(
        directory = val_dir,
        batch_size = batch_size,
        target_size = (IMG_SHAPE, IMG_SHAPE),
        shuffle = True,
        class_mode = 'categorical')

    # Test - flow from folder. 
    test_data_gen = datagen.flow_from_directory(
        directory = test_dir,
        batch_size = batch_size,
        target_size = (IMG_SHAPE, IMG_SHAPE),
        shuffle = False,
        class_mode = 'categorical')

    # FIT MODEL
    # Establish early stopping function to prevent overfitting.
    early_stop = EarlyStopping(
        monitor = 'val_loss', 
        patience = 5,  
        mode = 'min', 
        restore_best_weights = True)

    # Get history.
    H = model.fit(
        train_data_gen,
        validation_data = val_data_gen,
        epochs = 50,
        callbacks = [early_stop])

    # Plot history. 
    plot = plot_history(H, len(H.history['loss']))

    # Make predictions. 
    predictions = model.predict(
        test_data_gen, 
        batch_size = batch_size)

    # Preprocess labels for evaluation. 
    y_true = test_data_gen.classes
    y_pred = np.argmax(
        predictions, 
        axis = 1)
    class_labels = list(test_data_gen.class_indices.keys())

    # Make classification report. 
    report = classification_report(
        y_true, 
        y_pred, 
        target_names = class_labels)

    outpath = os.path.join(
            "out",
            "report.txt")

    with open(outpath, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()