import math
import matplotlib.pyplot as plt
import numpy as np
import random

class Complex:
    def __init__(self, real=0, imag=0):
        self._real = real
        self._imag = imag

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    @real.setter
    def real(self, value):
        if type(value) != float:
            raise TypeError('value must be float')
        else:
            self._real = value

    @imag.setter
    def imag(self, value):
        if type(value) != float:
            raise TypeError('value must be float')
        else:
            self._imag = value

    def __str__(self):
        s = ''

        if self._real == 0 and self._imag == 0:
            s += '0'
        elif self._real == 0 and self._imag != 0:
            if self._imag == 1:
                s += 'i'
            elif self._imag == -1:
                s += '-i'
            else:
                s += '{}i'.format(str(round(self._imag, 3)))
        elif self._real != 0 and self._imag == 0:
            s += str(round(self._real, 3))
        else:
            s += str(round(self._real, 3))
            
            if self._imag > 0:
                if self._imag == 1:
                    s += ' + i'
                else:
                    s += ' + {}i'.format(round(self._imag, 3))
            else:
                if self._imag == -1:
                    s += ' - i'
                else:
                    s += ' - {}i'.format(round(-self._imag, 3))

        return s

    def __repr__(self):
        return self.__str__()

    def modulus(self):
        return math.sqrt(self._real ** 2 + self._imag ** 2)

    def conjugate(self):
        return Complex(self._real, -self._imag)

    def arg(self):
        if self._real > 0 and self._imag > 0:
            return math.atan(self._imag / self._real)
        elif self._real < 0 and self._imag > 0:
            return math.pi + math.atan(self._imag / self._real)
        elif self._real > 0 and self._imag < 0:
            return 2 * math.pi + math.atan(self._imag / self._real)
        elif self._real < 0 and self._imag < 0:
            return math.pi + math.atan(self._imag / self._real)
        elif self._real > 0 and self._imag == 0:
            return 0
        elif self._real < 0 and self._imag == 0:
            return math.pi
        elif self._real == 0 and self._imag > 0:
            return math.pi / 2
        elif self._real == 0 and self._imag < 0:
            return 3 * math.pi / 2
        else:
            return 0

    def trigform(self):
        return '{}(cos({}) + i*sin({}))'.format(round(self.modulus(), 3), \
            round(self.arg(), 3), round(self.arg(), 3))

    def __round__(self, ndigits=0):
        real = round(self._real, ndigits)
        imag = round(self._imag, ndigits)

        return Complex(real, imag)

    def __abs__(self):
        real = abs(self._real)
        imag = abs(self._imag)

        return Complex(real, imag)

    def __getitem__(self, index):
        if index == 0 or index == -2:
            return self._real
        elif index == 1 or index == -1:
            return self._imag
        else:
            raise IndexError('index out of range')

    def __eq__(self, z):
        if isinstance(z, Complex):
            return self._real == z.real and self._imag == z.imag
        else:
            if self._imag == 0:
                return self._real == z
            else:
                return False

    def __ne__(self, z):
        if isinstance(z, Complex):
            return self._real != z.real and self._imag != z.imag
        else:
            if self._imag == 0:
                return self._real != z
            else:
                return True

    def __neg__(self):
        return Complex(-self._real, -self._imag)

    def __add__(self, z):
        if isinstance(z, Complex):
            real = self._real + z.real
            imag = self._imag + z.imag
        else:
            real = self._real + z
            imag = self._imag

        return Complex(real, imag)

    def __radd__(self, z):
        if isinstance(z, Complex):
            real = self._real + z.real
            imag = self._imag + z.imag
        else:
            real = self._real + z
            imag = self._imag

        return Complex(real, imag)

    def __iadd__(self, z):
        if isinstance(z, Complex):
            self._real += z.real
            self._imag += z.imag
        else:
            self._real += z

        return self

    def __sub__(self, z):
        if isinstance(z, Complex):
            real = self._real - z.real
            imag = self._imag - z.imag
        else:
            real = self._real - z
            imag = self._imag

        return Complex(real, imag)

    def __rsub__(self, z):
        if isinstance(z, Complex):
            real = z.real - self._real
            imag = self._imag
        else:
            real = z - self._real
            imag = self._imag

        return Complex(real, imag)

    def __isub__(self, z):
        if isinstance(z, Complex):
            self._real -= z.real
            self._imag -= z.imag
        else:
            self._real -= z

        return self

    def __mul__(self, z):
        if isinstance(z, Complex):
            real = self._real * z.real - self._imag * z.imag
            imag = self._real * z.imag + self._imag * z.real
        else:
            real = self._real * z
            imag = self._imag * z

        return Complex(real, imag)

    def __rmul__(self, z):
        if isinstance(z, Complex):
            real = self._real * z.real - self._imag * z.imag
            imag = self._real * z.imag + self._imag * z.real
        else:
            real = self._real * z
            imag = self._imag * z

        return Complex(real, imag)

    def __imul__(self, z):
        if isinstance(z, Complex):
            self._real = self._real * z.real - self._imag * z.imag
            self._imag = self._real * z.imag + self._imag * z.real
        else:
            self._real *= z
            self._imag *= z

        return self

    def __truediv__(self, z):
        if isinstance(z, Complex):
            res = (1 / z.modulus() ** 2) * self * z.conjugate()
        else:
            res = 1 / z * self

        return res

    def __rtruediv__(self, z):
        if isinstance(z, Complex):
            res = (1 / z.modulus() ** 2) * self * z.conjugate()
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
        self._real = r * math.cos(phi)
        self._imag = r * math.sin(phi)

    def __str__(self):
        return super.__str__()

    def __repr__(self):
        return super.__repr__()

def qfourier(coeffs):
    N = len(coeffs)
    transformed = []

    for k in range(N):
        bk = (1 / math.sqrt(N)) * Complex.csum([coeffs[j] * ComplexTrig( \
            1, 2 * math.pi * j * k / N) for j in range(N)])

        transformed.append(bk)

    return transformed

class Qubit:
    def __init__(self, coeffs):
        (bits_superpos, coeffs_len) = self.__qlen(len(coeffs))

        self._bits_superpos = bits_superpos
        self._coeffs_len = coeffs_len
        self._coeffs = coeffs

        if len(coeffs) < self._coeffs_len:
            for i in range(self._coeffs_len - len(coeffs)):
                self._coeffs.append(Complex(0, 0))

        self.normalize()

    @property
    def bits_superpos(self):
        return self._bits_superpos

    @property
    def coeffs_len(self):
        return self._coeffs_len

    @property
    def coeffs(self):
        return self._coeffs

    def __str__(self):
        s = ''

        for i in range(self._coeffs_len):
            if self._coeffs[i] != 0:
                if self._coeffs[i].real in (1, -1) and self._coeffs[i].imag == 0:
                    if self._coeffs[i].real == -1 and s.strip() == '':
                        s += f'-|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real > 0 and self._coeffs[i].imag == 0:
                    s += f'{self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real < 0 and self._coeffs[i].imag == 0:
                    if s.strip() == '':
                        s += f'{self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'{-self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real == 0 and self._coeffs[i].imag in (1, -1):
                    if self._coeffs[i].imag == -1 and s.strip() == '':
                        s += f'-i|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'i|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real == 0 and self._coeffs[i].imag > 0:
                    s += f'{self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real == 0 and self._coeffs[i].imag < 0:
                    if s.strip() == '':
                        s += f'{self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'{-self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                else:   
                    s += f'({self._coeffs[i]})|{i:0{self.bits_superpos}b}>'

            if i != self._coeffs_len - 1 and self._coeffs[i + 1] != 0 and s.strip() != '':
                if self._coeffs[i + 1].real > 0 and self._coeffs[i + 1].imag == 0:
                    s += ' + '
                elif self._coeffs[i + 1].real < 0 and self._coeffs[i + 1].imag == 0:
                    s += ' - '
                elif self._coeffs[i + 1].real == 0 and self._coeffs[i + 1].imag > 0:
                    s += ' + '
                elif self._coeffs[i + 1].real == 0 and self._coeffs[i + 1].imag < 0:
                    s += ' - '
                else:
                    s += ' + '

        return s

    def __len__(self):
        return self._coeffs_len

    def __getitem__(self, index):
        return self._coeffs[index]

    def normalize(self):
        norm = math.sqrt(sum(list(map(lambda c: c.modulus() ** 2, self._coeffs))))
        self._coeffs = list(map(lambda c: c / norm, self._coeffs))

    def measure(self):
        random_value = random.random()

        current_sum = 0
        for i in range(self._coeffs_len):
            current_sum += self._coeffs[i].modulus() ** 2

            if current_sum > random_value:
                print(f'Measured state: |{i:0{self._bits_superpos}b}>')

                new_coeffs = [Complex(0, 0) for j in range(i)] + [Complex(1, 0)] + \
                    [Complex(0, 0) for j in range(i + 1, self._coeffs_len)]
                self._coeffs = new_coeffs
                break

    def tensproduct(self, q):
        new_coeffs = []

        for i in range(self._coeffs_len):
            for j in range(len(q)):
                new_coeffs.append(self._coeffs[i] * q.coeffs[j])

        return Qubit(new_coeffs)

    def __qlen(self, list_len):
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

def main():
    pass

if __name__ == '__main__':
    main()


