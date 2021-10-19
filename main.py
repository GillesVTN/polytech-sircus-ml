import pandas as pd
import numpy as np

from models.heatmap import HeatMapBuilder
from models.csv_builder import CsvBuilder

PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

# builder = HeatMapBuilder()

# heatmap = builder.generate_heatmap(group="Patients", file_name="P_0011_73_M_JT.xlsx", sheet_name="mosob01.jpg - G")

# builder.print_heatmap(heatmap)

if __name__ == '__main__':
    # csv_builder = CsvBuilder()

    # csv_builder.generate_csv(jobs_nb=10)

    heatmap_builder = HeatMapBuilder()