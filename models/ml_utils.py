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
    Load data from the csv files and return list of heatmaps with associated labels
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

            # heatmap.core /= np.amax(heatmap.core) # Standard normalization based on matrix's max value

            # Normalization based on the number of measure equivalent to the max time elapsed
            heatmap.core /= heatmap.image_measures_number

            heatmaps.append((heatmap.core, heatmap_label))

        #np.random.shuffle(heatmaps)

    return [heatmap[0] for heatmap in heatmaps], [heatmap[1] for heatmap in heatmaps]


def split_datas(train_percentage: float, cores, labels):
    """
    Split the data into a tuple of training and test data based on a given ratio
    :param train_percentage: ratio of training data
    :param cores:
    :param labels:
    :return:
    """
    if len(cores) != len(labels):
        raise ValueError("Lenght of cores and labels not equals")
    split_number = int(len(cores) * train_percentage)

    return ((cores[:split_number]), labels[:split_number]), (cores[split_number:], labels[split_number:])