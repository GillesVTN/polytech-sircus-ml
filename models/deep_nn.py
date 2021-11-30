import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .heatmap import HeatmapBuilder, Heatmap
from .csv_builder import CsvBuilder

# get data from created CSV files

HEATMAP_SHAPE = (64, 64)

def display(images1, images2):
    n = 10
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def load_data():
    """
    Charge les données des csv et retour une liste d'heatmaps et une liste des labels associés
    :return:
    """
    heat_map_builder = HeatmapBuilder(HEATMAP_SHAPE)
    csv_builder = CsvBuilder()

    files_list = csv_builder.find_all_datas()

    heatmaps = []
    # heatmap_core = []
    # heatmap_label = []

    for file_name in files_list:
        file_name = "output/{}.csv".format(file_name.rsplit(".", 1)[0].rsplit("/", 1)[1])
        heatmaps_subject = heat_map_builder.generate_all_heatmaps_from_file(file_name)

        for heatmap in heatmaps_subject:
            # heatmap_core.append(heatmap.core)
            # TODO Normaliser les donnees de chaque heatmap
            if heatmap.label == "C":
                heatmap_label = 0
            else:
                heatmap_label = 1
            heatmap.core /= np.amax(heatmap.core)
            heatmaps.append((heatmap.core, heatmap_label))

        np.random.shuffle(heatmaps)

    return [heatmap[0] for heatmap in heatmaps], [heatmap[1] for heatmap in heatmaps]


def split_datas(train_percentage: float, cores, labels):
    """
    Sépare les données en un tuple données d'apprentissage et un tuple données de test selon un ratio
    :param train_percentage: pourcentage
    :param cores:
    :param labels:
    :return:
    """
    if len(cores) != len(labels):
        raise ValueError("Lenght of cores and labels not equals")
    split_number = int(len(cores) * train_percentage)

    return ((cores[:split_number]), labels[:split_number]), (cores[split_number:], labels[split_number:])

def learning(images, labels):
    # data gathering
    images = np.asarray(images)
    labels = np.asarray(labels)
    # split into train & test datasets
    (train_images, train_labels), (test_images, test_labels) = split_datas(0.8, images, labels)

    # create ml model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(64, 64)),
        tf.keras.layers.Normalization(axis=1),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    model.fit(train_images, train_labels, epochs=30)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print("Test accuracy: ", test_acc)


def learning_cnn(images, labels):
    """
    Modéle CNN
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
    # create ml model
    model = tf.keras.Sequential([
        # Partie convultionnelle
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

        # Classification
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print("Test accuracy: ", test_acc)


def learning_autoencoder(images, labels):
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

    autoencoder.fit(train_images, train_images, epochs=1, batch_size=128, shuffle=True, validation_data=(test_images, test_images))

    predictions = autoencoder.predict(test_images)

    #display(test_images, predictions)

    print(test_images[0].shape)
    plt.imshow(test_images[0], cmap="hot", interpolation="nearest")