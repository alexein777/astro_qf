from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import random

class Complex:
    def __init__(self, real=0, imag=0):
        self._real = real
        self._imag = imag
        self._real_rounded = round(self._real, 3)
        self._imag_rounded = round(self._imag, 3)

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    @property
    def real_rounded(self):
        return self._real_rounded

    @property
    def imag_rounded(self):
        return self._imag_rounded

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

        if self._real_rounded == 0 and self._imag_rounded == 0:
            s += '0'
        elif self._real_rounded == 0 and self._imag_rounded != 0:
            if self._imag_rounded == 1:
                s += 'i'
            elif self._imag_rounded == -1:
                s += '-i'
            else:
                s += '{}i'.format(str(self._imag_rounded))
        elif self._real_rounded != 0 and self._imag_rounded == 0:
            s += str(round(self._real_rounded))
        else:
            s += str(round(self._real_rounded))
            
            if self._imag_rounded > 0:
                if self._imag_rounded == 1:
                    s += ' + i'
                else:
                    s += ' + {}i'.format(str(self._imag_rounded))
            else:
                if self._imag_rounded == -1:
                    s += ' - i'
                else:
                    s += ' - {}i'.format(str(-self._imag_rounded))

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
        return '{}*(cos({}) + i*sin({}))'.format(round(self.modulus(), 3), \
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
            return self._real != z.real or self._imag != z.imag
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
        real = self._real + z
        imag = self._imag
            
        return Complex(real, imag)

    def __iadd__(self, z):
        if isinstance(z, Complex):
            self._real += z.real
            self._imag += z.imag
            self._real_rounded = round(self._real, 3)
            self._imag_rounded = round(self._imag, 3)
        else:
            self._real += z
            self._real_rounded = round(self._real, 3)

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
        real = z - self._real
        imag = -self._imag

        return Complex(real, imag)

    def __isub__(self, z):
        if isinstance(z, Complex):
            self._real -= z.real
            self._imag -= z.imag
            self._real_rounded = round(self._real, 3)
            self._imag_rounded = round(self._imag, 3)
        else:
            self._real -= z
            self._real_rounded = round(self._real, 3)

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
        real = self._real * z
        imag = self._imag * z

        return Complex(real, imag)

    def __imul__(self, z):
        if isinstance(z, Complex):
            self._real = self._real * z.real - self._imag * z.imag
            self._imag = self._real * z.imag + self._imag * z.real
            self._real_rounded = round(self._real, 3)
            self._imag_rounded = round(self._imag, 3)
        else:
            self._real *= z
            self._imag *= z
            self._real_rounded = round(self._real, 3)
            self._imag_rounded = round(self._imag, 3)

        return self

    def __truediv__(self, z):
        if isinstance(z, Complex):
            res = (1 / z.modulus() ** 2) * self * z.conjugate()
        else:
            res = 1 / z * self

        return res

    def __rtruediv__(self, z):
        return z * self.conjugate() / (self.modulus() ** 2)

    def __pow__(self, k=5):
        arg = self.arg()
        mod = self.modulus()

        real = mod ** k * math.cos(k * arg)
        imag = mod ** k * math.sin(k * arg)

        return Complex(real, imag)

    def sin(self):
        real = math.sin(self.real) * math.cosh(self.imag)
        imag = math.cos(self.real) * math.sinh(self.imag)

        return Complex(real, imag)

    def cos(self):
        real = math.cos(self.real) * math.cosh(self.imag)
        imag = math.sin(self.real) * math.sinh(self.imag)

        return Complex(real, -imag)

    @classmethod
    def csum(self, list_complex):
        z = Complex()

        for i in range(len(list_complex)):
            z += list_complex[i]

        return z

    @classmethod
    def csin(self, list_complex):
        return list(map(lambda x: x.sin(), list_complex))

    @classmethod
    def ccos(self, list_complex):
        return list(map(lambda x: x.cos(), list_complex))

    @classmethod
    def random(self, start=-10, stop=10):
        real = (stop - start) * random.random() + start
        imag = (stop - start) * random.random() + start

        return Complex(real, imag)

    @classmethod
    def randlist(self, size=50, start=-10, stop=10):
        return [Complex.random(start, stop) for i in range(size)]

    @classmethod
    def plot(self, list_complex, color_string='bo', label_str='', marker_size=None):
        if type(list_complex) == list:
            plt.plot([c.real for c in list_complex], [c.imag for c in list_complex], color_string, label=label_str, markersize=marker_size)
        else:
            raise TypeError('plot accepts list of complex numbers (class Complex)')

    @classmethod
    def create_circle(self, center=(0, 0), radius=1, num=50):
        return [Complex(center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi)) \
            for phi in np.linspace(0, 2 * np.pi, num, endpoint=False)]

    @classmethod
    def create_square(self, center=(0, 0), edge=1, num=48):
        bottom_left = (center[0] - edge / 2, center[1] - edge / 2)
        b_x = bottom_left[0]
        b_y = bottom_left[1]

        dots_per_edge = num // 4
        # mid_dots = (num - 4) // 4
        edge_step = edge / dots_per_edge 

        corners = [Complex(b_x, b_y), Complex(b_x + edge, b_y), \
            Complex(b_x, b_y + edge), Complex(b_x + edge, b_y + edge)]
        bottom = [Complex(x, b_y) for x in \
            [x1 for x1 in np.arange(b_x + edge_step, b_x + edge, edge_step)]]
        top = [Complex(x, b_y + edge) for x in \
            [x1 for x1 in np.arange(b_x + edge_step, b_x + edge, edge_step)]]
        left = [Complex(b_x, y) for y in \
            [y1 for y1 in np.arange(b_y + edge_step, b_y + edge, edge_step)]]
        right = [Complex(b_x + edge, y) for y in \
            [y1 for y1 in np.arange(b_y + edge_step, b_y + edge , edge_step)]]
        remain = []

        if (num % 4 != 0):
            diff = num % 4

            for i in range(diff):
                rand_edge = random.choice(['bottom', 'top', 'left', 'right'])

                if rand_edge == 'bottom':
                    rand_x = b_x + edge * random.random()
                    c = Complex(rand_x, b_y)
                elif rand_edge == 'top':
                    rand_x = b_x + edge * random.random()
                    c = Complex(rand_x, b_y + edge)
                elif rand_edge == 'left':
                    rand_y = b_y + edge * random.random()
                    c = Complex(b_x, rand_y)
                else:
                    rand_y = b_y + edge * random.random()
                    c = Complex(b_x + edge, rand_y)

                remain.append(c)
                    
        return corners + bottom + top + left + right + remain

class ComplexTrig(Complex):
    def __init__(self, phi, r=1):
        self._real = r * math.cos(phi)
        self._imag = r * math.sin(phi)
        self._real_rounded = round(self._real, 3)
        self._imag_rounded = round(self._imag, 3)

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
    
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
                if self._coeffs[i].real_rounded in (1, -1) and self._coeffs[i].imag_rounded == 0:
                    if self._coeffs[i].real_rounded == -1 and s.strip() == '':
                        s += f'-|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real_rounded > 0 and self._coeffs[i].imag_rounded == 0:
                    s += f'{self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real_rounded < 0 and self._coeffs[i].imag_rounded == 0:
                    if s.strip() == '':
                        s += f'{self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'{-self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real_rounded == 0 and self._coeffs[i].imag_rounded in (1, -1):
                    if self._coeffs[i].imag_rounded == -1 and s.strip() == '':
                        s += f'-i|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'i|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real_rounded == 0 and self._coeffs[i].imag_rounded > 0:
                    s += f'{self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                elif self._coeffs[i].real_rounded == 0 and self._coeffs[i].imag_rounded < 0:
                    if s.strip() == '':
                        s += f'{self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                    else:
                        s += f'{-self._coeffs[i]}|{i:0{self.bits_superpos}b}>'
                else:   
                    s += f'({self._coeffs[i]})|{i:0{self.bits_superpos}b}>'

            if i != self._coeffs_len - 1 and self._coeffs[i + 1] != 0 and s.strip() != '':
                if self._coeffs[i + 1].real_rounded > 0 and self._coeffs[i + 1].imag_rounded == 0:
                    s += ' + '
                elif self._coeffs[i + 1].real_rounded < 0 and self._coeffs[i + 1].imag_rounded == 0:
                    s += ' - '
                elif self._coeffs[i + 1].real_rounded == 0 and self._coeffs[i + 1].imag_rounded > 0:
                    s += ' + '
                elif self._coeffs[i + 1].real_rounded == 0 and self._coeffs[i + 1].imag_rounded < 0:
                    s += ' - '
                else:
                    s += ' + '

        return s

    def __repr__(self):
        return self.__str__()

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

        return self

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

    @classmethod
    def random(self, coeffs_size=2):
        return Qubit(Complex.randlist(coeffs_size))

    @classmethod
    def qrandom(self, list_size=20, qubit_size=2):
        return [Qubit.random(qubit_size) for i in range(list_size)]

    @classmethod
    def plot(self, qubit, color_string='bo', label_str='', marker_size=None):
        if isinstance(qubit, Qubit):
            Complex.plot(qubit.coeffs, color_string, label_str, marker_size)
        elif type(qubit) == list:
            Complex.plot(qubit, color_string, label_str, marker_size)
        else:
            raise TypeError('plot accepts Qubit or list type')

class QubitBloch(Qubit):
    def __init__(self, theta=0, phi=0, coeffs=None):
        if coeffs != None:
            if type(coeffs) == list:
                if len(coeffs) < 2:
                    raise ValueError('incorrect coeffs length for qubit on Bloch\'s sphere (expected 2)')
                else:
                    self._theta = 2 * math.acos(coeffs[0].real)
                    self._phi = coeffs[1].arg()

                    self._bits_superpos = 1
                    self._coeffs_len = 2
                    self._coeffs = coeffs[:2]
            else:
                raise ValueError('expected list of Complex coeffs, given {}'.format(type(coeffs)))
        else:
            self._theta = theta
            self._phi = phi
            self._bits_superpos = 1
            self._coeffs_len = 2
    
            coeffs = [Complex(math.cos(theta / 2)), ComplexTrig(phi) * math.sin(theta / 2)]
            self._coeffs = coeffs

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__str__()

    @property
    def theta(self):
        return self._theta

    @property
    def phi(self):
        return self._phi

    @classmethod
    def random(self):
        theta = math.pi * random.random()
        phi = 2 * math.pi * random.random()
        
        return QubitBloch(theta, phi)

    @classmethod
    def randlist(self, num=50):
        return [QubitBloch.random() for i in range(num)]

    @classmethod
    def create_sphere(self, num=100):
        sphere = []
        q_root = int(math.floor(math.sqrt(num)))

        for theta in np.linspace(0, math.pi, q_root):
            for phi in np.linspace(0, 2 * math.pi, q_root):
                sphere.append(QubitBloch(theta, phi))

        if (q_root ** 2 < num):
            for i in range(num - q_root ** 2):
                theta = math.pi * random.random()
                phi = 2 * math.pi * random.random()

                sphere.append(QubitBloch(theta, phi))

        return sphere

    @classmethod
    def qscatter(self, qubits, axis, colorq='b', markerq='o', labelq=''):
        if type(qubits) == list:
            xs = list(map(lambda q: math.cos(q.phi) * math.sin(q.theta), qubits))
            ys = list(map(lambda q: math.sin(q.phi) * math.sin(q.theta), qubits))
            zs = list(map(lambda q: math.cos(q.theta), qubits))

            axis.scatter(xs, ys, zs, c=colorq, marker=markerq, label=labelq)
        else:
            raise TypeError('qscatter accepts list of qubits (class QubitBloch)')

def qfourier(list_param):
    transformed = []

    if isinstance(list_param, (Qubit, QubitBloch)):
        N = list_param.coeffs_len
    
        for k in range(N):
            bk = (1 / math.sqrt(N)) * Complex.csum([list_param.coeffs[j] * \
                ComplexTrig(2 * math.pi * j * k / N) for j in range(N)])
    
            transformed.append(bk)
    elif type(list_param) == list:
        if len(list_param) > 0:
            if isinstance(list_param[0], Complex):
                N = len(list_param)
        
                for k in range(N):
                    bk = (1 / math.sqrt(N)) * Complex.csum([list_param[j] * \
                        ComplexTrig(2 * math.pi * j * k / N) for j in range(N)])
        
                    transformed.append(bk)
            elif isinstance(list_param[0], QubitBloch):
                transformed = list(map(lambda q: QubitBloch(coeffs=qfourier(q.coeffs)), list_param))
            elif isinstance(list_param[0], Qubit):
                transformed = list(map(lambda q: Qubit(qfourier(q.coeffs)), list_param))
            else:
                raise TypeError('unsupported object type for QFT')
        else:
            raise ValueError('empty list of coeffs')
    else:
        raise TypeError('unsupported collection type for QFT')

    return transformed