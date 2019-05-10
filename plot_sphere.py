from mpl_toolkits.mplot3d import Axes3D
from quantum_fourier import *
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    fig = plt.figure()
    fig_size = plt.gcf()
    fig_size.set_size_inches(12, 6)

    # qubit Bloch
    ax = fig.add_subplot(121, projection='3d')

    qubit_sphere = QubitBloch.create_sphere(num=576)
    QubitBloch.qscatter(qubit_sphere, ax, labelq='qubit on Bloch\'s sphere')

    ax.set_zlim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
# 
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    plt.legend(loc='upper right', fontsize='small')

    # transformed qubit Bloch
    ax_t = fig.add_subplot(122, projection='3d')
    qsphere_transformed = qfourier(qubit_sphere)
    QubitBloch.qscatter(qsphere_transformed, ax_t, colorq='r', markerq='^', labelq='transformed qubit on Bloch\'s sphere')

    ax_t.set_zlim(-1, 1)
    ax_t.set_xlim(-1, 1)
    ax_t.set_ylim(-1, 1)
# 
    ax_t.set_xlabel('x-axis')
    ax_t.set_ylabel('y-axis')
    ax_t.set_zlabel('z-axis')

    plt.legend(loc='upper right', fontsize='small')
# 
    plt.show()

if __name__ == '__main__':
    main()