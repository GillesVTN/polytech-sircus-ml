import numpy as np
import tensorflow as tf

from .heatmap import HeatmapBuilder, Heatmap
from .csv_builder import CsvBuilder

# get data from created CSV files


def load_data():
    heat_map_builder = HeatmapBuilder()
    csv_builder = CsvBuilder()

    files_list = csv_builder.find_all_datas()

    heatmaps = []
    #heatmap_core = []
    #heatmap_label = []

    for file_name in files_list:
        file_name = "output/{}.csv".format(file_name.rsplit(".", 1)[0].rsplit("/", 1)[1])
        heatmaps_subject = heat_map_builder.generate_all_heatmaps_from_file(file_name)

        for heatmap in heatmaps_subject:
            #heatmap_core.append(heatmap.core)
            if heatmap.label == "C":
                heatmap_label = 0
            else:
                heatmap_label = 1

            heatmaps.append((heatmap.core, heatmap_label))

        np.random.shuffle(heatmaps)

    return [heatmap[0] for heatmap in heatmaps], [heatmap[1] for heatmap in heatmaps]

def split_datas(train_percentage:float, cores, labels):
    """

    :param train_percentage:
    :param cores:
    :param labels:
    :return:
    """
    if len(cores) != len(labels):
        raise ValueError("Lenght of cores and labels not equals")
    split_number = int(len(cores) * train_percentage)

    return (cores[:split_number], labels[:split_number]), (cores[split_number:], labels[split_number:])


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
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    model.fit(train_images, train_labels, epochs=30)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


    print("Test accuracy: ", test_acc)

