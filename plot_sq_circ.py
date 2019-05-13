from quantum_fourier import *
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    fig_size = plt.gcf()
    fig_size.set_size_inches(12, 6)

    plt.figure(1)

    # square, transformed square
    square = Complex.create_square(edge=3, num=64)
    square_transformed = qfourier(square)

    plt.subplot(121)
    plt.xlabel('real')
    plt.ylabel('imag')
    Complex.plot(square, label_str='complex square', marker_size=3)
    Complex.plot(square_transformed, 'r^', label_str='transformed complex square', marker_size=3)

    plt.legend()

    # circle, transformed circle
    circle = Complex.create_circle(radius=1.5, num=40, center=(1.5, 1.5))
    circle_transformed = qfourier(circle)

    # for c in circle_transformed:
    #     print('({}, {})'.format(c.real, c.imag))

    plt.subplot(122)
    plt.xlabel('real')
    plt.ylabel('imag')
    Complex.plot(circle, label_str='complex circle', marker_size=2)
    Complex.plot(circle_transformed, 'r^', label_str='transformed complex circle', marker_size=2)

    plt.legend()
    plt.xlim(-2, 11)
    plt.ylim(-2, 11)

    plt.show()

if __name__ == '__main__':
    main()
