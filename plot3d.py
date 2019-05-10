from mpl_toolkits.mplot3d import Axes3D
from quantum_fourier import Complex, Qubit, qfourier
import matplotlib.pyplot as plt
import numpy as np

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = np.arange(-5, 5, 0.2)
    ys = np.arange(-5, 5, 0.2)

    ax.scatter(xs, ys)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    plt.show()

if __name__ == '__main__':
    main()