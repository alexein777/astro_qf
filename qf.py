import math

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

    def __getitem__(self, index):
        if index == 0 or index == -2:
            return self.real
        elif index == 1 or index == -1:
            return self.imag
        else:
            raise IndexError('Index error: index out of range')

    def __eq__(self, z):
        if isinstance(z, Complex):
            return self.real == z.getReal() and self.imag == z.getImag()
        else:
            return self.real == z

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
        mod = z.modulus()

        if isinstance(z, Complex):
            res = 1 / mod * self * z.conjugate()
        else:
            res = 1 / mod * self

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

def qfourier(coeffs):
    N = len(coeffs)
    transformed = []

    for k in range(N):
        bk = (1 / math.sqrt(N)) * Complex.csum([ComplexTrig( \
            coeffs[j], 2 * math.pi * j * k / N) for j in range(N)])
        # bk = (1 / math.sqrt(N)) * Complex.csum([Complex(coeffs[j] * \
        #     math.cos(2 * math.pi * j * k / N), coeffs[j] * math.sin(2 * math.pi * j * k / N)) for j in range(N)])
        transformed.append(bk)

    return transformed
