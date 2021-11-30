import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Heatmap:

    def __init__(self, shape: tuple, label: str):
        self.core = np.zeros(shape)
        self.label = label

    def print(self):
        """
        Affiche une fenetre contenant l'image de l'heatmap
        :return:
        """
        plt.imshow(self.core, cmap='hot', interpolation='nearest')
        plt.axis("off")
        plt.show()

    def save_to_png(self, path_and_filename: str):
        """
        Enregistre l'image de la heatmap en png
        :param path_and_filename:
        :return:
        """
        plt.imshow(self.core, cmap='hot', interpolation='nearest')
        plt.axis("off")
        plt.savefig(path_and_filename)


class HeatmapBuilder:
    def __init__(self, main_path="Acquisitions_Eye_tracking_objets_visages_Fix_Seq1",
                 heatmap_shape=(64, 64), images_shape=(1000, 1000)):
        self.main_path = main_path
        self.heatmap_shape = heatmap_shape
        self.images_shape = images_shape

    """
    def read_file(self, file_name, sheet_name):
        data = pd.read_excel("{}/{}.csv".format(self.main_path, file_name), sheet_name=sheet_name, header=1)

        return data"""

    def generate_heatmap(self, file_data: pd.DataFrame, image_name: str, label: str) -> Heatmap:
        """
        Génére une heatmap à partir d'un fichier csv et  de son image
        :param file_data:
        :param image_name:
        :param label:
        :return:
        """

        image_data = file_data[file_data["image_name"] == image_name]
        # print(image_data)
        heatmap = Heatmap(self.heatmap_shape, label)

        # Get number of measures for future normalization
        heatmap.image_measures_number = len(image_data)

        for index, row in image_data.iterrows():
            # print(image_name, row['x'], row['y'])
            h_x = int((self.heatmap_shape[0] - 1) * (row['x'] / self.images_shape[0])) - 1
            h_y = int((self.heatmap_shape[1] - 1) * (row['y'] / self.images_shape[1])) - 1
            # print(h_x, h_y)
            heatmap.core[h_x][h_y] += 1

        return heatmap

    def generate_all_heatmaps_from_file(self, file_name: str) -> [Heatmap]:
        """
        Génére toutes les heatmaps à partir d'un fichier csv
        :param file_name:
        :return:
        """
        file_data = pd.read_csv(file_name)

        images_names_list = file_data["image_name"].unique()
        heatmaps = []

        for image_name in images_names_list:
            heatmaps.append(self.generate_heatmap(file_data, image_name, file_name.split("output/", 1)[1][0]))

        return heatmaps
