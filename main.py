import pandas as pd
import numpy as np

from models.heatmap import HeatmapBuilder
from models.csv_builder import CsvBuilder
from models.deep_nn import *
import matplotlib.pyplot as plt

PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

if __name__ == '__main__':
    # csv_builder = CsvBuilder()

    # csv_builder.generate_csv(jobs_nb=10)

    # heatmap_builder = HeatmapBuilder()

    learning()