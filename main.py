from models import ml_utils, cnn, nn, vgg, vgg_classif, csv_builder, heatmap, autoencoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd


# PATH_FILES = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"

class Classifiers():
    def __init__(self):
        self.images = None
        self.labels = None

    def generate_heatmaps(self):
        builder = csv_builder.CsvBuilder()
        builder.generate_csv(jobs_nb=10)

    def load_heatmaps(self):
        if self.images is None and self.labels is None:
            print("Loading heatmaps...")
            self.images, self.labels = ml_utils.load_data()
            print("done.")

    def learn_fcn(self, n_epochs):
        self.load_heatmaps()
        return nn.learn(self.images, self.labels, n_epochs)

    def learn_cnn(self, n_epochs):
        self.load_heatmaps()
        return cnn.learn_cnn(self.images, self.labels, n_epochs)

    def predict_vgg(self):
        self.load_heatmaps()
        vgg.predict_vgg(self.images, self.labels)

    def learn_classfif_vgg(self, n_epochs):
        return vgg_classif.learn(n_epochs)

    def learn_autoencoder(self, n_epochs):
        self.load_heatmaps()
        autoencoder.learn_autoencoder(self.images, self.labels)

    def plot_history(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def plot_pca(self):
        data = np.genfromtxt("output_vgg_features.txt")

        pca = PCA(n_components=3)
        components = pca.fit_transform(data)
        print(pca.explained_variance_ratio_)
        total_var = pca.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            components, x=0, y=1, z=2,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show()


if __name__ == '__main__':
    classifiers = Classifiers()

    #classifiers.learn_cnn(50)

    classifiers.learn_classfif_vgg(n_epochs=20)
