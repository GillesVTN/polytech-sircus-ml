

# PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

if __name__ == '__main__':
    # csv_builder = CsvBuilder()

    # csv_builder.generate_csv(jobs_nb=10)

    # heatmap_builder = HeatmapBuilder()

    images, labels = ml_utils.load_data()

    #print(labels)

    cnn.learn_cnn(images, labels, n_epochs=20)
    #autoencoder.learn_autoencoder(images, labels)
