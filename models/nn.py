import numpy as np
import tensorflow as tf

from .ml_utils import split_datas

def nn_model():
    """
    Neural network model
    :return: neural network model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(64, 64)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.summary()

    return model

def learn(images, labels, n_epochs=10):
    """
    Apprentissage du réseau nn
    :param images:
    :param labels:
    :param n_epochs: nombre d'epochs
    :return:
    """

    # data gathering
    images = np.asarray(images)
    labels = np.asarray(labels)
    # split into train & test datasets
    (train_images, train_labels), (test_images, test_labels) = split_datas(0.8, images, labels)

    # create ml model
    model = nn_model()

    history = model.fit(train_images, train_labels, epochs=n_epochs, validation_split=0.1)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print("Test accuracy: ", test_acc)

    return history