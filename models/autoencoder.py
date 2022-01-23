import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from .ml_utils import split_datas, display
from sklearn.metrics import accuracy_score

def learn_autoencoder(images, labels):
    """
    Créer le modèle d'auto-encodeur, et l'entraîne avec nos données
    :param images:
    :param labels:
    :return:
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

    print(train_images.shape)
    print(test_images.shape)

    """
    Design de l'autoencodeur

    Entrée		64x64x1
    Conv 8		64x64x8
    MaxPool 2	32x32x8
    Conv 16 	32x32x16
    MaxPool 2	16x16x16
    Conv 32 	16x16x32
    MaxPool 2	8x8x32
    Conv 64 	8x8x64
    MaxPool 2	4x4x64
    Conv 128	4x4x128
    MaxPool 2	2x2x128
    Conv 256	2x2x256
    MaxPool 2	1x1x256

    Milieu		1x1x256

    Conv 256	    1x1x256
    UpSampling 2	2x2x256
    Conv 128	    2x2x128
    UpSampling 2	4x4x128
    Conv 64		    4x4x64
    UpSampling 2	8x8x64
    Conv 32		    8x8x32
    UpSampling 2	16x16x32
    Conv 16		    16x16x16
    UpSampling 2	32x32x16
    Conv 8		    64x64x8
    Conv 1		    64x64x1	sigmoid
    """

    input_img = tf.keras.Input(shape=(64, 64, 1))  # 64x64x1
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(input_img)  # 64x64x8
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)  # 32x32x8
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)  # 32x32x16
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)  # 16x16x16
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)  # 16x16x32
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 8x8x32
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)  # 8x8x64
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 4x4x64
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)  # 4x4x128
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 2x2x128
    x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)  # 2x2x256
    encoded = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 1x1x256

    x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(encoded)  # 1x1x256
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 2x2x256
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)  # 2x2x128
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 4x4x128
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)  # 4x4x64
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 8x8x64
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)  # 8x8x32
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 16x16x32
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)  # 16x16x16
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 32x32x16
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)  # 32x32x8
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 64x64x8
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)  # 64x64x1

    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer="rmsprop", loss="binary_crossentropy")
    autoencoder.summary()

    # plot the autoencoder
    #plot_model(autoencoder, 'autoencoder_compress.png', show_shapes=True)

    autoencoder.fit(train_images, train_images, epochs=1, batch_size=128, shuffle=True,
                    validation_data=(test_images, test_images))

    predictions = autoencoder.predict(test_images)

    # calculate classification accuracy
    acc = accuracy_score(test_labels, predictions)
    print(acc)
    #display(test_images, predictions)

    print(test_images[0].shape)
    plt.imshow(test_images[0], cmap="hot", interpolation="nearest")