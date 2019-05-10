from mpl_toolkits.mplot3d import Axes3D
from quantum_fourier import Complex, Qubit, qfourier
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = []
    ys = []
    zs = []

    points = 700
    r = 1

    p_root = int(math.floor(math.sqrt(points)))
    # for phi in np.arange(0, 2 * np.pi, np.pi / 20):
    #     for theta in np.arange(0, np.pi, np.pi / 20):
    for phi in np.linspace(0, 2 * np.pi, p_root):
        for theta in np.linspace(0, np.pi, p_root):
            xs.append(r * np.cos(phi) * np.sin(theta))
            ys.append(r * np.sin(phi) * np.sin(theta))
            zs.append(r * np.cos(theta))

    ax.set_zlim(-1, 1)
    ax.scatter(xs, ys, zs)
    
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    plt.show()

if __name__ == '__main__':
    main()