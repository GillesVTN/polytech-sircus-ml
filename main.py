from models import ml_utils, cnn, nn, vgg, vgg_classif, autoencoder
import numpy as np
import pandas as pd

# PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

if __name__ == '__main__':


    # csv_builder = CsvBuilder()
    # csv_builder.generate_csv(jobs_nb=10)
    # heatmap_builder = HeatmapBuilder()

    images, labels = ml_utils.load_data()
    autoencoder.learn_autoencoder(images, labels)
    #print(labels)

    #vgg.predict_vgg(images, labels)
