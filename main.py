import pandas as pd
import numpy as np

from models.heatmap import HeatmapBuilder
from models.csv_builder import CsvBuilder
from models.deep_nn import load_data, split_datas
import matplotlib.pyplot as plt


PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

if __name__ == '__main__':
    # csv_builder = CsvBuilder()

    # csv_builder.generate_csv(jobs_nb=10)

    #heatmap_builder = HeatmapBuilder()
    (cores, labels) = load_data()
    (train_cores, train_labels), (test_cores, test_labels) = split_datas(80, cores, labels)
    print(len(train_cores))
    print(len(test_cores))
