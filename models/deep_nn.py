import tensorflow as tf

from .heatmap import HeatmapBuilder, Heatmap
from .csv_builder import CsvBuilder

# get data from created CSV files


def load_data():
    heat_map_builder = HeatmapBuilder()
    csv_builder = CsvBuilder()

    files_list = csv_builder.find_all_datas()

    heatmap_core = []
    heatmap_label = []

    for file_name in files_list:
        file_name = "output/{}.csv".format(file_name.rsplit(".", 1)[0].rsplit("/", 1)[1])
        heatmaps_subject = heat_map_builder.generate_all_heatmaps_from_file(file_name)
        for heatmap in heatmaps_subject:
            heatmap_core.append(heatmap.core)
            heatmap_label.append(heatmap.label)

    return heatmap_core, heatmap_label

def split_datas(train_purcentage:int, cores, labels):
    """

    :param train_purcentage:
    :param cores:
    :param labels:
    :return:
    """
    if len(cores) != len(labels):
        raise ValueError("Lenght of cores and labels not equals")
    split_number = int(len(cores)*(train_purcentage/100))

    return (cores[:split_number], labels[:split_number]), (cores[split_number:], labels[split_number:])


def create_model():

    images, labels = load_data()

    #images = [images[k] /= max(images[k] for k)]

    (train_images, train_labels), (test_images, test_labels) = split_datas(80, images, labels)

