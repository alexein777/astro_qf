from quantum_fourier import Complex, Qubit, qfourier
import matplotlib.pyplot as plt

def main():
    complex_li = Complex.randlist(50)
    complex_transformed = qfourier(complex_li)

    q = Qubit.random(16)
    q_transformed = Qubit(qfourier(q))

    fig_size = plt.gcf()
    # fig_size.set_size_inches(7, 12)
    fig_size.set_size_inches(12, 6)

    plt.figure(1)

    # complex, transformed complex
    plt.subplot(121)
    plt.xlabel('real')
    plt.ylabel('imag')
    Complex.plot(complex_li, label_str='complex number', marker_size=5)
    Complex.plot(complex_transformed, 'r^', 'transformed complex number')
    plt.legend(loc='upper center', fontsize='small', bbox_to_anchor=(0.5, 1.111))

    # qubit, transformed qubit
    plt.subplot(122)
    plt.xlabel('real')
    plt.ylabel('imag')
    Qubit.plot(q, label_str='initial qubit')
    Qubit.plot(q_transformed, 'r^', 'transformed qubit')
    plt.legend(loc='upper center', fontsize='small', bbox_to_anchor=(0.5, 1.111))

    plt.show()

if __name__ == '__main__':
    main()
