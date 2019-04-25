import math
import matplotlib.pyplot as plt
import numpy as np
import random

class Complex:
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag
    
    def __str__(self):
        s = ''

        if self.real == 0 and self.imag == 0:
            s += '0'
        elif self.real == 0 and self.imag != 0:
            if self.imag == 1:
                s += 'i'
            elif self.imag == -1:
                s += '-i'
            else:
                s += '{}i'.format(str(round(self.imag, 3)))
        elif self.real != 0 and self.imag == 0:
            s += str(round(self.real, 3))
        else:
            s += str(round(self.real, 3))
            
            if self.imag > 0:
                if self.imag == 1:
                    s += ' + i'
                else:
                    s += ' + {}i'.format(round(self.imag, 3))
            else:
                if self.imag == -1:
                    s += ' - i'
                else:
                    s += ' - {}i'.format(round(-self.imag, 3))

        return s

    def __repr__(self):
        return self.__str__()

    def getReal(self):
        return self.real

    def getImag(self):
        return self.imag

    def setReal(self, real):
        self.real = real

    def setImag(self, imag):
        self.imag = imag

    def modulus(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def conjugate(self):
        return Complex(self.real, -self.imag)

    def arg(self):
        if self.real > 0 and self.imag > 0:
            return math.atan(self.imag / self.real)
        elif self.real < 0 and self.imag > 0:
            return math.pi + math.atan(self.imag / self.real)
        elif self.real > 0 and self.imag < 0:
            return 2 * math.pi + math.atan(self.imag / self.real)
        elif self.real < 0 and self.imag < 0:
            return math.pi + math.atan(self.imag / self.real)
        elif self.real > 0 and self.imag == 0:
            return 0
        elif self.real < 0 and self.imag == 0:
            return math.pi
        elif self.real == 0 and self.imag > 0:
            return math.pi / 2
        elif self.real == 0 and self.imag < 0:
            return 3 * math.pi / 2
        else:
            return 0

    def trigform(self):
        return '{}(cos({}) + i*sin({}))'.format(round(self.modulus(), 3), \
            round(self.arg(), 3), round(self.arg(), 3))

    def __round__(self, ndigits=0):
        real = round(self.real, ndigits)
        imag = round(self.imag, ndigits)

        return Complex(real, imag)

    def __abs__(self):
        real = abs(self.real)
        imag = abs(self.imag)

        return Complex(real, imag)

    def __getitem__(self, index):
        if index == 0 or index == -2:
            return self.real
        elif index == 1 or index == -1:
            return self.imag
        else:
            raise IndexError('index out of range')

    def __eq__(self, z):
        if isinstance(z, Complex):
            return self.real == z.getReal() and self.imag == z.getImag()
        else:
            if self.imag == 0:
                return self.real == z
            else:
                return False

    def __ne__(self, z):
        if isinstance(z, Complex):
            return self.real != z.getReal() and self.imag != z.getImag()
        else:
            if self.imag == 0:
                return self.real != z
            else:
                return True

    def __neg__(self):
        return Complex(-self.real, -self.imag)

    def __add__(self, z):
        if isinstance(z, Complex):
            real = self.real + z.getReal()
            imag = self.imag + z.getImag()
        else:
            real = self.real + z
            imag = self.imag

        return Complex(real, imag)

    def __radd__(self, z):
        if isinstance(z, Complex):
            real = self.real + z.getReal()
            imag = self.imag + z.getImag()
        else:
            real = self.real + z
            imag = self.imag

        return Complex(real, imag)

    def __iadd__(self, z):
        if isinstance(z, Complex):
            self.real += z.getReal()
            self.imag += z.getImag()
        else:
            self.real += z

        return self

    def __sub__(self, z):
        if isinstance(z, Complex):
            real = self.real - z.getReal()
            imag = self.imag - z.getImag()
        else:
            real = self.real - z
            imag = self.imag

        return Complex(real, imag)

    def __rsub__(self, z):
        if isinstance(z, Complex):
            real = z.getReal() - self.real
            imag = self.imag
        else:
            real = z - self.real
            imag = self.imag

        return Complex(real, imag)

    def __isub__(self, z):
        if isinstance(z, Complex):
            self.real -= z.getReal()
            self.imag -= z.getImag()
        else:
            self.real -= z

        return self

    def __mul__(self, z):
        if isinstance(z, Complex):
            real = self.real * z.getReal() - self.imag * z.getImag()
            imag = self.real * z.getImag() + self.imag * z.getReal()
        else:
            real = self.real * z
            imag = self.imag * z

        return Complex(real, imag)

    def __rmul__(self, z):
        if isinstance(z, Complex):
            real = self.real * z.getReal() - self.imag * z.getImag()
            imag = self.real * z.getImag() + self.imag * z.getReal()
        else:
            real = self.real * z
            imag = self.imag * z

        return Complex(real, imag)

    def __imul__(self, z):
        if isinstance(z, Complex):
            self.real = self.real * z.getReal() - self.imag * z.getImag()
            self.imag = self.real * z.getImag() + self.imag * z.getReal()
        else:
            self.real *= z
            self.imag *= z

        return self

    def __truediv__(self, z):
        if isinstance(z, Complex):
            res = 1 / z.modulus() * self * z.conjugate()
        else:
            res = 1 / z * self

        return res

    def __rtruediv__(self, z):
        if isinstance(z, Complex):
            mod = z.modulus()
            res = (1 / mod ** 2) * self * z.conjugate()
        else:
            res = z * self.conjugate() / (self.modulus() ** 2)

        return res

    def __pow__(self, k):
        arg = self.arg()
        mod = self.modulus()

        real = mod ** k * math.cos(k * arg)
        imag = mod ** k * math.sin(k * arg)

        return Complex(real, imag)

    @classmethod
    def csum(self, list_complex):
        z = Complex()

        for i in range(len(list_complex)):
            z += list_complex[i]

        return z

class ComplexTrig(Complex):
    def __init__(self, r, phi):
        self.real = r * math.cos(phi)
        self.imag = r * math.sin(phi)

    def __str__(self):
        return super.__str__()

    def __repr__(self):
        return super.__repr__()

# Napomena: ne radi ispis elemenata ubacenih u transformed zbog poziva funkcije round
# u okviru __str__ metoda
def qfourier(coeffs):
    N = len(coeffs)
    transformed = []

    for k in range(N):
        bk = (1 / math.sqrt(N)) * Complex.csum([coeffs[j] * ComplexTrig( \
            1, 2 * math.pi * j * k / N) for j in range(N)])


        print(bk)
        transformed.append(bk)

    return transformed

class Qubit:
    def __init__(self, coeffs):
        (bits_superpos, coeffs_len) = self._qlen(len(coeffs))

        self.bits_superpos = bits_superpos
        self.coeffs_len = coeffs_len
        self.coeffs = coeffs

        if len(coeffs) < self.coeffs_len:
            for i in range(self.coeffs_len - len(coeffs)):
                self.coeffs.append(Complex(0, 0))

        self.normalize()

    def __str__(self):
        s = ''

        for i in range(self.coeffs_len):
            if self.coeffs[i] != 0:
                if self.coeffs[i].getReal() in (1, -1) and self.coeffs[i].getImag() == 0:
                    if self.coeffs[i].getReal() == -1 and s.strip() == '':
                        s += f'-|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'|{i:0{self.bits_superpos}b}>'
                elif self.coeffs[i].getReal() > 0 and self.coeffs[i].getImag() == 0:
                    s += f'{self.coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self.coeffs[i].getReal() < 0 and self.coeffs[i].getImag() == 0:
                    if s.strip() == '':
                        s += f'{self.coeffs[i]}|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'{-self.coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self.coeffs[i].getReal() == 0 and self.coeffs[i].getImag() in (1, -1):
                    if self.coeffs[i].getImag() == -1 and s.strip() == '':
                        s += f'-i|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'i|{i:0{self.bits_superpos}b}>'
                elif self.coeffs[i].getReal() == 0 and self.coeffs[i].getImag() > 0:
                    s += f'{self.coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self.coeffs[i].getReal() == 0 and self.coeffs[i].getImag() < 0:
                    if s.strip() == '':
                        s += f'{self.coeffs[i]}|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'{-self.coeffs[i]}|{i:0{self.bits_superpos}b}>'
                else:   
                    s += f'({self.coeffs[i]})|{i:0{self.bits_superpos}b}>'

            if i != self.coeffs_len - 1 and self.coeffs[i + 1] != 0 and s.strip() != '':
                if self.coeffs[i + 1].getReal() > 0 and self.coeffs[i + 1].getImag() == 0:
                    s += ' + '
                elif self.coeffs[i + 1].getReal() < 0 and self.coeffs[i + 1].getImag() == 0:
                    s += ' - '
                elif self.coeffs[i + 1].getReal() == 0 and self.coeffs[i + 1].getImag() > 0:
                    s += ' + '
                elif self.coeffs[i + 1].getReal() == 0 and self.coeffs[i + 1].getImag() < 0:
                    s += ' - '
                else:
                    s += ' + '

        return s

    def __len__(self):
        return self.coeffs_len

    def getCoeffs(self):
        return self.coeffs

    def getBits(self):
        return self.bits_superpos

    def normalize(self):
        norm = math.sqrt(sum(list(map(lambda c: c.modulus() ** 2, self.coeffs))))
        self.coeffs = list(map(lambda c: c / norm, self.coeffs))

    def measure(self):
        random_value = random.random()

        current_sum = 0
        for i in range(self.coeffs_len):
            current_sum += self.coeffs[i].modulus() ** 2

            if current_sum > random_value:
                print(f'Measured state: |{i:0{self.bits_superpos}b}>')

                new_coeffs = [Complex(0, 0) for j in range(i)] + [Complex(1, 0)] + [Complex(0, 0) for j in range(i + 1, self.coeffs_len)]
                self.coeffs = new_coeffs
                break

    def tensproduct(self, q):
        q_coeffs = q.getCoeffs()
        new_coeffs = []

        for i in range(self.coeffs_len):
            for j in range(len(q)):
                new_coeffs.append(self.coeffs[i] * q_coeffs[j])

        return Qubit(new_coeffs)

    def _qlen(self, list_len):
        n = 0
        deg2 = True

        while list_len != 0:
            if list_len != 1 and list_len % 2 != 0:
                deg2 = False

            list_len //= 2
            n += 1

        if not deg2:
            return n, 2 ** n
        else:
            return n - 1, 2 ** (n - 1)

# plt.plot(list(map(lambda x: math.sin(x), list(np.arange(0, 4*math.pi, 0.01)))))
# plt.plot(list(map(lambda x: math.cos(x), list(np.arange(0, 4*math.pi, 0.01)))))
# plt.show()
