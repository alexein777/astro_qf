from quantum_fourier import Complex, Qubit, qfourier
import matplotlib.pyplot as plt

def main():
    complex_li = Complex.crandom(50)
    complex_transformed = qfourier(complex_li)

    q = Qubit.random(16)
    q_transformed = Qubit(qfourier(q))

    fig_size = plt.gcf()
    fig_size.set_size_inches(7, 12)

    # Reprezentacija komplkesnih brojeva
    plt.figure(1)
    plt.subplot(211)
    plt.ylabel('imag')
    Complex.plot(complex_li, label_str='complex numbers')
    Complex.plot(complex_transformed, 'ro', 'transformed complex numbers')
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.1, 1.1))

    # Reprezentacija kjubita
    plt.subplot(212)
    plt.xlabel('real')
    plt.ylabel('imag')
    Qubit.plot(q, label_str='initial qubit')
    Qubit.plot(q_transformed, 'ro', 'transformed qubit')
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.1, 1.1))

    plt.show()

if __name__ == '__main__':
    main()
