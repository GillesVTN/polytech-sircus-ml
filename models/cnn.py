import numpy as np
import tensorflow as tf

from .ml_utils import split_datas

def cnn_model():
    """
    Créer le modèle CNN
    :return: modèle CNN Keras
    """

    model = tf.keras.Sequential([
        # Partie convultionnelle
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classification
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    return model

def learn_cnn(images, labels, n_epochs):
    """
    Apprentisage du modèle  CNN
    :param images:
    :param labels:
    :return: test_acc, history
    """
    # data gathering
    images = np.asarray(images)
    labels = np.asarray(labels)

    # split into train & test datasets
    (train_images, train_labels), (test_images, test_labels) = split_datas(0.8, images, labels)

    train_images = np.array(train_images)
    train_images = train_images.reshape((train_images.shape[0], 64, 64, 1))
    train_images = train_images.astype("float32")

    test_images = np.array(test_images)
    test_images = test_images.reshape((test_images.shape[0], 64, 64, 1))

    # create ml model
    model = cnn_model()

    history = model.fit(train_images, train_labels, epochs=n_epochs, validation_split=0.1)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print("Test accuracy: ", test_acc)

    return history
