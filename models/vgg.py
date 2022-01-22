import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Model

from .ml_utils import split_datas


def nn_model_classif():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(224, activation="relu", input_shape=(None,4096)),
        tf.keras.layers.Dense(112, activation="relu"),
        tf.keras.layers.Dense(56, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.summary()
    return model


def predict_vgg(images, labels, n_epochs=10):
    # data gathering
    images = np.asarray(images)
    labels = np.asarray(labels)
    # split into train & test datasets
    (train_images, train_labels), (test_images, test_labels) = split_datas(1., images, labels)

    # load the model
    model = VGG16()
    # load an image from file
    images_arr = train_images
    print("Base:", images_arr.shape)

    # upscale image
    images_arr = np.resize(images_arr, (images_arr.shape[0], 224, 224))
    print("apres resize:", images_arr.shape)
    # matrix to 3 rgb matrix
    images_arr = np.stack((images_arr,) * 3, axis=-1)
    print("apres rgb:", images_arr.shape)
    # reshape data for the model
    images_arr = images_arr.reshape(
        (images_arr.shape[0], images_arr.shape[1], images_arr.shape[2], images_arr.shape[3]))
    print("apres reshape:", images_arr.shape)
    # prepare the image for the VGG model
    images_arr = preprocess_input(images_arr)
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    # predict the probability across all output classes
    yhat = model.predict(images_arr)
    print(yhat)
    print(yhat.shape)
    np.savetxt("output_vgg_features_shuffle.txt", yhat)
    np.savetxt("output_labels_shuffle.txt", train_labels)

    """
    label = model.predict(train_images)

    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2] * 100))
    """
