
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.linalg import multi_dot
import math
import sys

def covariance_matrix(samples):
    dimension = samples.shape[1]
    size = len(samples)

    total = np.zeros([dimension], dtype=np.float)

    for i in range(dimension):
        sum = 0
        for j in range(size):
            sum += samples[j][i]

        f = 9
        total[i] = sum


    mean = total / size

    matrix = np.empty([dimension, dimension], dtype=np.float)

    for i in range(dimension):
        for j in range(dimension):
            element = 0
            for k in range(size):
                element += (samples[k][i] - mean[i]) * (samples[k][j] - mean[j])
                matrix[i, j] = element / size
    return matrix

def main():
    samples1 = np.array([[2, 6, 7, 13], [4, 6, 3, 44], [3, 4, 11, 5], [3, 8, 6, 1], [1, 22, 7, 9]], dtype=np.float)
    samples2 = np.array([[1, -2], [5, -2], [3, 0], [3, -4]], dtype=np.int16)
    mean = covariance_matrix(samples1)

if __name__ == '__main__':
    main()