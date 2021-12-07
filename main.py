from models import heatmap, nn, ml_utils

#from .models import nn, cnn, autoencoder

# PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

if __name__ == '__main__':
    # csv_builder = CsvBuilder()

    # csv_builder.generate_csv(jobs_nb=10)

    # heatmap_builder = HeatmapBuilder()

    images, labels = ml_utils.load_data()

    #print(labels)
    nn.learn(images, labels)
    #learning_cnn(images, labels)
    #autoencoder.learn_autoencoder(images, labels)