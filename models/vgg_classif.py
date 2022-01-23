import numpy as np
import tensorflow as tf

from .ml_utils import split_datas

def nn_model_classif():
    """
    Modele servant à la classification des donneés issues de vgg16
    :return: keras modèle
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(4096,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    return model


def learn(features_file_name="output_vgg_features_shuffle.txt", labels_file_name="output_labels_shuffle.txt", n_epochs=20):

    # Separation train/test 80/20 %
    features = np.loadtxt(features_file_name)
    labels = np.loadtxt(labels_file_name)
    (train_features, train_labels), (test_features, test_labels) = split_datas(0.8, features, labels)
    #train_features = train_features.reshape((train_features.shape[0], 4096))
    #test_features = test_features.reshape((test_features.shape[0], 4096))
    print(train_features.shape)

    model_classif = nn_model_classif()

    history = model_classif.fit(train_features, train_labels, epochs=n_epochs, shuffle=True, validation_split=0.1)

    test_loss, test_acc = model_classif.evaluate(test_features, test_labels, verbose=2)

    print("Test accuracy: ", test_acc)
    print("Test loss: ", test_loss)

    return history