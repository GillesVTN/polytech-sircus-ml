import tensorflow as tf

from heatmap import *
from csv_builder import CsvBuilder

# get data from created CSV files


def load_data():
    heat_map_builder = HeatmapBuilder()
    csv_builder = CsvBuilder()

    files_list = csv_builder.find_all_datas()

    for file_name in files_list:
        heat_map_builder.generate_all_heatmaps_from_file(file_name)