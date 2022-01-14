from models import ml_utils, cnn, nn, vgg
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

# PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

if __name__ == '__main__':
    # csv_builder = CsvBuilder()

    # csv_builder.generate_csv(jobs_nb=10)

    # heatmap_builder = HeatmapBuilder()

    images, labels = ml_utils.load_data()

    #print(labels)

    vgg.predict_vgg(images, labels)

    #data = np.genfromtxt("output_vgg_features.txt")

    #pca = PCA(n_components=3)
    #components = pca.fit_transform(data)
    #print(pca.explained_variance_ratio_)
    #total_var = pca.explained_variance_ratio_.sum() * 100

    '''
    fig = px.scatter_3d(
        components, x=0, y=1, z=2,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()'''