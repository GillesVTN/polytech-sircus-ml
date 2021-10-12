import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class HeatMapBuilder:
    def __init__(self, main_path="Acquisitions_Eye_tracking_objets_visages_Fix_Seq1",
                 heatmap_shape=(64, 64), images_shape=(1000, 1000)):
        self.main_path = main_path
        self.heatmap_shape = heatmap_shape
        self.images_shape = images_shape

    def read_file(self, group, file_name, sheet_name):
        data = pd.read_excel("{}/{}/{}".format(self.main_path, group, file_name), sheet_name=sheet_name, header=1)

        return data

    def generate_heatmap(self, group, file_name, sheet_name):
        """

        :param group:
        :param file_name:
        :param sheet_name:
        :return: HeatMap
        """
        data = self.read_file(group, file_name, sheet_name)

        heatmap = HeatMap(core=np.zeros(self.heatmap_shape))

        for index, row in data.iterrows():
            print(row['x'], row['y'])
            h_x = round(self.heatmap_shape[0] * (row['x'] / self.images_shape[0]))
            h_y = round(self.heatmap_shape[1] * (row['y'] / self.images_shape[1]))
            print(h_x, h_y)
            heatmap.core[h_x][h_y] += 1

        return heatmap

    def generate_all(self):
        pass

class HeatMap:

    def __init__(self, core):
        self.core = core

    def print_heatmap(self, heatmap):
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()

    def save_to_(self):
        pass