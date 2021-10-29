from models.deep_nn import *

# PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

if __name__ == '__main__':
    # csv_builder = CsvBuilder()

    # csv_builder.generate_csv(jobs_nb=10)

    # heatmap_builder = HeatmapBuilder()

    images, labels = load_data()

    #print(labels)

    learning(images, labels)

