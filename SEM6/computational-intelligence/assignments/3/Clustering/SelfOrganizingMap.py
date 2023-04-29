import json
from warnings import warn

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap


class SOM(object):

    def __init__(self,
                 x: int,
                 y: int,
                 input_len: int,
                 sigma: int | float = 1.0,
                 learning_rate: int | float = 0.5) -> None:
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = np.random.RandomState()

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len

        self._weights = self._random_generator.rand(x, y, input_len) * 2 - 1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = np.zeros((x, y))
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)

        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

        self._decay_function = lambda learning_rate, t, max_iter: learning_rate / (
            1 + t / (max_iter / 2))
        self.neighborhood = self.gaussian
        self._activation_distance = self.euclidean_distance

    def activate(self, x: np.ndarray) -> None:
        self._activation_map = self._activation_distance(x, self._weights)

    def gaussian(self, c: int, sigma: int | float) -> np.ndarray:
        d = 2 * sigma * sigma
        ax = np.exp(-np.power(self._xx - self._xx.T[c], 2) / d)
        ay = np.exp(-np.power(self._yy - self._yy.T[c], 2) / d)
        return (ax * ay).T

    def euclidean_distance(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.subtract(x, w), axis=-1)

    def check_input_len(self, data: np.ndarray) -> None:
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x: np.ndarray) -> tuple[int, int]:
        self.activate(x)
        return np.unravel_index(self._activation_map.argmin(),
                                self._activation_map.shape)

    def update(self, x: np.ndarray, win: int, t: int,
               max_iteration: int) -> None:
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        sig = self._decay_function(self._sigma, t, max_iteration)
        g = self.neighborhood(win, sig) * eta
        self._weights += np.einsum('ij, ijk->ijk', g, x - self._weights)

    def random_weights_init(self, data: np.ndarray) -> None:
        self.check_input_len(data)
        it = np.nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            it.iternext()

    def train(self, data: np.ndarray, num_iteration: int) -> None:
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')
        self.check_input_len(data)
        iterations = self.build_iteration_indexes(len(data), num_iteration,
                                                  self._random_generator)

        for t, iteration in enumerate(iterations):
            self.update(data[iteration], self.winner(data[iteration]), t,
                        num_iteration)

    @staticmethod
    def build_iteration_indexes(data_len, num_iterations, random_generator):
        iterations = np.arange(num_iterations) % data_len
        random_generator.shuffle(iterations)
        return iterations


class Support:

    def __init__(self, map_size=(10, 10), sigma=1.0, learning_rate=0.5):
        self.map_size = map_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.som = None
        self.cluster_labels = None
        self.cluster_colors = None

    def load_data(self, data):
        self.data = data
        year = 1950
        self.data = self.data[self.data['Time'] == year]
        self.countries = self.data['Location']

        self.lons = []
        self.lats = []

        with open('country-codes-lat-long-alpha3.json') as f:
            data = json.load(f)
            for country in self.countries:
                for item in data['ref_country_codes']:
                    if item['country'] == country:
                        self.lons.append(item['longitude'])
                        self.lats.append(item['latitude'])
                        break
                else:
                    self.data = self.data[self.data['Location'] != country]

        self.data = self.data.select_dtypes(exclude=['object'])
        self.data_norm = (self.data - self.data.min()) / (self.data.max() -
                                                          self.data.min())
        self.input_len = len(self.data_norm.columns)

    def train(self, n_iterations=10000):
        if self.data is None:
            raise Exception('Data has not been loaded.')
        self.som = SOM(self.map_size[0],
                       self.map_size[1],
                       self.input_len,
                       sigma=self.sigma,
                       learning_rate=self.learning_rate)
        self.som.random_weights_init(self.data_norm.values)
        self.som.train(self.data_norm.values, n_iterations)

    def cluster(self):
        if self.som is None:
            raise Exception('SOM has not been trained.')
        self.cluster_labels = np.zeros(len(self.data_norm))
        self.cluster_colors = []
        for i in range(len(np.unique(self.cluster_labels))):
            x = self.data_norm.values[i]
            c = self.som.winner(x)
            self.cluster_labels[i] = c[0] * self.som._weights.shape[1] + c[1]
            self.cluster_colors.append((c[0] / self.som._weights.shape[0],
                                        c[1] / self.som._weights.shape[1], 0.5))

    def visualize_map(self, feature1=None, feature2=None):
        if self.som is None:
            raise Exception('SOM has not been trained.')
        if feature1 is None:
            raise Exception('Feature 1 has not been specified.')
        if feature2 is None:
            raise Exception('Feature 2 has not been specified.')
        if self.cluster_labels is None:
            raise Exception('Data has not been clustered.')
        fig = plt.figure(figsize=(16, 12))
        m = Basemap(projection='robin', resolution='l', lat_0=0, lon_0=0)
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.fillcontinents(color='lightgray', lake_color='white')
        m.drawmapboundary(fill_color='white')

        x, y = m(self.lons, self.lats)
        m.scatter(x, y, s=100, c=self.cluster_colors, marker='o', alpha=0.5)

        plt.show()


som = Support()

data = pd.read_csv('WPP2019_TotalPopulationBySex.csv')

som.load_data(data)

som.train()

som.cluster()

som.visualize_map(feature1='PopMale', feature2='PopDensity')