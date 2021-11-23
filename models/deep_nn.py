import numpy as np
import tensorflow as tf

from .heatmap import HeatmapBuilder, Heatmap
from .csv_builder import CsvBuilder

# get data from created CSV files

HEATMAP_SIZE = (64, 64)


def load_data():
    heat_map_builder = HeatmapBuilder(HEATMAP_SIZE)
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

    :param train_percentage:
    :param cores:
    :param labels:
    :return:
    """
    if len(cores) != len(labels):
        raise ValueError("Lenght of cores and labels not equals")
    split_number = int(len(cores) * train_percentage)

    return ((cores[:split_number]), labels[:split_number]), (cores[split_number:], labels[split_number:])


def fully_connectec_model():
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

    return model


def learning(images, labels):
    # data gathering
    images = np.asarray(images)
    labels = np.asarray(labels)
    # split into train & test datasets
    (train_images, train_labels), (test_images, test_labels) = split_datas(0.8, images, labels)

    # create ml model
    model = cnn_model()

    model.fit(train_images, train_labels, epochs=30)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print("Test accuracy: ", test_acc)


def learning_cnn(images, labels):
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

    input_img = tf.keras.Input(shape=(64, 64, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation="relu", padding="same")(x)

    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.summary()


    autoencoder.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels))
