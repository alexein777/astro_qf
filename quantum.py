from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import copy


def max_min_x_y(iterable):
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for it in iterable:
        if it[0] < min_x:
            min_x = it[0]
        if it[0] > max_x:
            max_x = it[1]
        if it[1] < min_y:
            min_y = it[1]
        if it[1] > max_y:
            max_y = it[1]

    return min_x, min_y, max_x, max_y


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def noise(amp=1, amp_scale=1, noise_freq=100):
    distort_apply = random.random()

    if distort_apply < noise_freq / 100:
        amp_percent = random.random()
        scale_factor = 2 * amp / amp_scale
        amp_distort = random.random() * amp * amp_percent / scale_factor
        sgn = random.choice([-1, 1])

        res = sgn * amp_distort
    else:
        res = 0

    return res


class Complex:
    def __init__(self, real=0, imag=0, ndigits=3):
        self._real = real
        self._imag = imag
        self._ndigits = ndigits
        self._real_rounded = round(self._real, ndigits)
        self._imag_rounded = round(self._imag, ndigits)

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
            s += str(self._real_rounded)
        else:
            s += str(self._real_rounded)

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
        return np.sqrt(self._real ** 2 + self._imag ** 2)

    def conjugate(self):
        return Complex(self._real, -self._imag)

    def arg(self):
        if self._real > 0 and self._imag > 0:
            return np.arctan(self._imag / self._real)
        elif self._real < 0 and self._imag > 0:
            return np.pi + np.arctan(self._imag / self._real)
        elif self._real > 0 and self._imag < 0:
            return 2 * np.pi + np.arctan(self._imag / self._real)
        elif self._real < 0 and self._imag < 0:
            return np.pi + np.arctan(self._imag / self._real)
        elif self._real > 0 and self._imag == 0:
            return 0
        elif self._real < 0 and self._imag == 0:
            return np.pi
        elif self._real == 0 and self._imag > 0:
            return np.pi / 2
        elif self._real == 0 and self._imag < 0:
            return 3 * np.pi / 2
        else:
            return 0

    def trigform(self):
        arg_deg = np.rad2deg(self.arg())

        return '{}*(cos({}) + i*sin({}))'.format(round(self.modulus(), self._ndigits),
                                                 round(arg_deg, self._ndigits),
                                                 round(arg_deg, self._ndigits))

    def __round__(self, ndigits=3):
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
            self._real_rounded = round(self._real, self._ndigits)
            self._imag_rounded = round(self._imag, self._ndigits)
        else:
            self._real += z
            self._real_rounded = round(self._real, self._ndigits)

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
            self._real_rounded = round(self._real, self._ndigits)
            self._imag_rounded = round(self._imag, self._ndigits)
        else:
            self._real -= z
            self._real_rounded = round(self._real, self._ndigits)

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
            self._real_rounded = round(self._real, self._ndigits)
            self._imag_rounded = round(self._imag, self._ndigits)
        else:
            self._real *= z
            self._imag *= z
            self._real_rounded = round(self._real, self._ndigits)
            self._imag_rounded = round(self._imag, self._ndigits)

        return self

    def __truediv__(self, z):
        if isinstance(z, Complex):
            res = (1 / z.modulus() ** 2) * self * z.conjugate()
        else:
            res = 1 / z * self

        return res

    def __rtruediv__(self, z):
        return z * self.conjugate() / (self.modulus() ** 2)

    def __pow__(self, k=2):
        arg = self.arg()
        mod = self.modulus()

        real = mod ** k * np.cos(k * arg)
        imag = mod ** k * np.sin(k * arg)

        return Complex(real, imag)

    def sin(self):
        real = np.sin(self.real) * np.cosh(self.imag)
        imag = np.cos(self.real) * np.sinh(self.imag)

        return Complex(real, imag)

    def cos(self):
        real = np.cos(self.real) * np.cosh(self.imag)
        imag = np.sin(self.real) * np.sinh(self.imag)

        return Complex(real, -imag)

    @classmethod
    def csum(cls, list_complex):
        z = Complex()

        for z_i in list_complex:
            z += z_i

        return z

    @classmethod
    def csin(cls, list_complex):
        return list(map(lambda x: x.sin(), list_complex))

    @classmethod
    def ccos(cls, list_complex):
        return list(map(lambda x: x.cos(), list_complex))

    @classmethod
    def random(cls, start=-10, stop=10):
        real = (stop - start) * random.random() + start
        imag = (stop - start) * random.random() + start

        return Complex(real, imag)

    @classmethod
    def randlist(cls, size=50, start=-10, stop=10):
        return [Complex.random(start, stop) for i in range(size)]

    @classmethod
    def unpack(cls, list_complex):
        if type(list_complex) == list or type(list_complex) == np.ndarray:
            reals, imags = [], []
            for z in list_complex:
                reals.append(z.real)
                imags.append(z.imag)

            return reals, imags
        else:
            raise TypeError('unpack accepts list or numpy array of complex numbers (class Complex)')

    @classmethod
    def plot(cls, list_complex, color_string='bo', label='', markersize=None):
        if type(list_complex) == list or type(list_complex) == np.ndarray:
            reals, imags = cls.unpack(list_complex)
            plt.plot(reals, imags, color_string, label=label, markersize=markersize)
        else:
            raise TypeError('plot accepts list or numpy array of complex numbers (class Complex)')

    @classmethod
    def scatter(cls, list_complex, s=2, c='b', marker='o', alpha=1.0):
        if type(list_complex) == list or type(list_complex) == np.ndarray:
            reals, imags = cls.unpack(list_complex)
            plt.scatter(reals, imags, s=s, c=c, marker=marker, alpha=alpha)
        else:
            raise TypeError('plot accepts list or numpy array of complex numbers (class Complex)')

    @classmethod
    def create_circle(cls, center=(0, 0), radius=1, points=50):
        return [Complex(center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi))
                for phi in np.linspace(0, 2 * np.pi, points, endpoint=False)]

    @classmethod
    def create_filled_circle(cls, center=(0, 0), radius=1, points_outer=50):
        filled_circle = [Complex(center[0], center[1])]
        radius_iter = radius
        radius_step = 10 * radius / points_outer
        num_iter = points_outer

        while radius_iter > 0 and num_iter > 0:
            filled_circle += Complex.create_circle(center=center, radius=radius_iter, points=num_iter)
            radius_iter -= radius_step
            num_iter -= np.floor(30 * radius_step)

        return filled_circle

    @classmethod
    def create_square(cls, center=(0, 0), edge=1, edge_points=48):
        bottom_left = (center[0] - edge / 2, center[1] - edge / 2)
        b_x = bottom_left[0]
        b_y = bottom_left[1]

        points_per_edge = edge_points // 4 + 1
        mid_points = (edge_points - 4) // 4

        if points_per_edge == 1:
            edge_step = edge / 2
        else:
            edge_step = edge / (points_per_edge - 1)

        corners = [Complex(b_x, b_y), Complex(b_x + edge, b_y), \
                   Complex(b_x, b_y + edge), Complex(b_x + edge, b_y + edge)]

        bottom = []
        top = []
        left = []
        right = []
        for i in range(mid_points):
            bottom.append(Complex(b_x + (i + 1) * edge_step, b_y))
            top.append(Complex(b_x + (i + 1) * edge_step, b_y + edge))
            left.append(Complex(b_x, b_y + (i + 1) * edge_step))
            right.append(Complex(b_x + edge, b_y + (i + 1) * edge_step))

        remain = []
        if edge_points % 4 != 0:
            diff = edge_points % 4

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

    @classmethod
    def create_filled_square(cls, center=(0, 0), edge=1, edge_points_outer=48):
        filled_square = []
        num_iter = edge_points_outer
        edge_iter = edge

        points_per_edge = edge_points_outer // 4 + 1
        edge_step = edge / (points_per_edge - 1)

        while edge_iter > 0 and num_iter > 0:
            filled_square += Complex.create_square(center, edge_iter, num_iter)
            edge_iter -= 2 * edge_step
            num_iter -= 8

        # Slucaj kada je centar prazan zbog uniformne raspodele tacaka po ivicama
        if edge_points_outer % 8 == 0:
            filled_square.append(Complex(center[0], center[1]))

        return filled_square

    @classmethod
    def create_cube_data(cls, center=(0, 0, 0), edge=1, points_per_side_edge=60):
        side = cls.create_filled_square(center=center, edge=edge, edge_points_outer=points_per_side_edge)

        n = len(side)
        bottom_plane_fix = [center[2] - edge / 2] * n
        top_plane_fix = [center[2] + edge / 2] * n
        back_plane_fix = [center[1] - edge / 2] * n
        front_plane_fix = [center[1] + edge / 2] * n
        left_plane_fix = [center[0] - edge / 2] * n
        right_plane_fix = [center[0] + edge / 2] * n

        return side, bottom_plane_fix, top_plane_fix, back_plane_fix, front_plane_fix, left_plane_fix, right_plane_fix

    @classmethod
    def plot_cube(cls, cube_data, ax, sizes=None, colors=None, markers=None, alphas=None):
        side, \
        bottom_plane_fix, \
        top_plane_fix, \
        back_plane_fix, \
        front_plane_fix, \
        left_plane_fix, \
        right_plane_fix \
            = cube_data

        reals, imags = cls.unpack(side)

        if sizes is None or len(sizes) < 6:
            sizes = [2] * 6
        if colors is None or len(colors) < 6:
            colors = ['blue'] * 6
        if markers is None or len(markers) < 6:
            markers = ['o'] * 6
        if alphas is None or len(alphas) < 6:
            alphas = [1.0] * 6

        s1, c1, m1, a1 = sizes[0], colors[0], markers[0], alphas[0]
        s2, c2, m2, a2 = sizes[1], colors[1], markers[1], alphas[1]
        s3, c3, m3, a3 = sizes[2], colors[2], markers[2], alphas[2]
        s4, c4, m4, a4 = sizes[3], colors[3], markers[3], alphas[3]
        s5, c5, m5, a5 = sizes[4], colors[4], markers[4], alphas[4]
        s6, c6, m6, a6 = sizes[5], colors[5], markers[5], alphas[5]

        ax.scatter(reals, imags, bottom_plane_fix, s=s1, c=c1, marker=m1, alpha=a1)
        ax.scatter(reals, imags, top_plane_fix, s=s2, c=c2, marker=m2, alpha=a2)

        ax.scatter(reals, back_plane_fix, imags, s=s3, c=c3, marker=m3, alpha=a3)
        ax.scatter(reals, front_plane_fix, imags, s=s4, c=c4, marker=m4, alpha=a4)

        ax.scatter(left_plane_fix, reals, imags, s=s5, c=c5, marker=m5, alpha=a5)
        ax.scatter(right_plane_fix, reals, imags, s=s6, c=c6, marker=m6, alpha=a6)

    @classmethod
    def create_triangle(cls, v1, v2, v3, edge_points=50):
        a = distance(v1, v2)
        b = distance(v1, v3)
        c = distance(v2, v3)

        o = a + b + c
        if o == 0.0:
            return []

        k = edge_points / o

        ppa = int(a * k) - 1
        ppb = int(b * k) - 1
        ppc = int(c * k) - 1

        if ppa < 1 or ppb < 1 or ppc < 1:
            return []

        v1v2_step_x = np.abs(v1[0] - v2[0]) / ppa
        v1v2_step_y = np.abs(v1[1] - v2[1]) / ppa
        v1v2 = []
        for i in range(1, ppa):
            x = v1[0] + i * v1v2_step_x if v1[0] < v2[0] else v1[0] - i * v1v2_step_x
            y = v1[1] + i * v1v2_step_y if v1[1] < v2[1] else v1[1] - i * v1v2_step_y
            v1v2.append(Complex(x, y))

        v1v3_step_x = np.abs(v1[0] - v3[0]) / ppb
        v1v3_step_y = np.abs(v1[1] - v3[1]) / ppb
        v1v3 = []
        for i in range(1, ppb):
            x = v1[0] + i * v1v3_step_x if v1[0] < v3[0] else v1[0] - i * v1v3_step_x
            y = v1[1] + i * v1v3_step_y if v1[1] < v3[1] else v1[1] - i * v1v3_step_y
            v1v3.append(Complex(x, y))

        v2v3_step_x = np.abs(v2[0] - v3[0]) / ppc
        v2v3_step_y = np.abs(v2[1] - v3[1]) / ppc
        v2v3 = []
        for i in range(1, ppc):
            x = v2[0] + i * v2v3_step_x if v2[0] < v3[0] else v2[0] - i * v2v3_step_x
            y = v2[1] + i * v2v3_step_y if v2[1] < v3[1] else v2[1] - i * v2v3_step_y
            v2v3.append(Complex(x, y))

        vertices = [Complex(v1[0], v1[1]), Complex(v2[0], v2[1]), Complex(v3[0], v3[1])]

        remain = []
        for i in range(edge_points - ppa - ppb - ppc):
            chosen_edge = np.random.randint(1, 4)
            if chosen_edge == 1:  # stranica a
                if v1[0] < v2[0]:
                    x = (v2[0] - v1[0]) * np.random.random() + v1[0]
                    y = v1[1] + (v2[1] - v1[1]) / (v2[0] - v1[0]) * (x - v1[0])
                elif v1[0] == v2[0]:
                    x = v1[0]
                    if v1[1] < v2[1]:
                        y = (v2[1] - v1[1]) * np.random.random() + v1[1]
                    else:
                        y = (v1[1] - v2[1]) * np.random.random() + v2[1]
                else:
                    x = (v1[0] - v2[0]) * np.random.random() + v2[0]
                    y = v1[1] + (v2[1] - v1[1]) / (v2[0] - v1[0]) * (x - v1[0])
            elif chosen_edge == 2:  # stranica b
                if v1[0] < v3[0]:
                    x = (v3[0] - v1[0]) * np.random.random() + v1[0]
                    y = v1[1] + (v3[1] - v1[1]) / (v3[0] - v1[0]) * (x - v1[0])
                elif v1[0] == v3[0]:
                    x = v1[0]
                    if v1[1] < v3[1]:
                        y = (v3[1] - v1[1]) * np.random.random() + v1[1]
                    else:
                        y = (v1[1] - v3[1]) * np.random.random() + v3[1]
                else:
                    x = (v1[0] - v3[0]) * np.random.random() + v3[0]
                    y = v1[1] + (v3[1] - v1[1]) / (v3[0] - v1[0]) * (x - v1[0])
            else:  # stranica c
                if v2[0] < v3[0]:
                    x = (v3[0] - v2[0]) * np.random.random() + v2[0]
                    y = v2[1] + (v3[1] - v2[1]) / (v3[0] - v2[0]) * (x - v2[0])
                elif v2[0] == v3[0]:
                    x = v2[0]
                    if v2[1] < v3[1]:
                        y = (v3[1] - v2[1]) * np.random.random() + v2[1]
                    else:
                        y = (v2[1] - v3[1]) * np.random.random() + v3[1]
                else:
                    x = (v2[0] - v3[0]) * np.random.random() + v3[0]
                    y = v2[1] + (v3[1] - v2[1]) / (v3[0] - v2[0]) * (x - v2[0])

            remain.append(Complex(x, y))

        return vertices + v1v2 + v1v3 + v2v3 + remain

    @classmethod
    def create_filled_triangle(cls, v1_outer, v2_outer, v3_outer, edge_points_outer=50):
        c = ((v1_outer[0] + v2_outer[0] + v3_outer[0]) / 3, (v1_outer[1] + v2_outer[1] + v3_outer[1]) / 3)  # teziste

        v1c = distance(v1_outer, c)
        v2c = distance(v2_outer, c)
        v3c = distance(v3_outer, c)

        vc_max = max([v1c, v2c, v3c])
        it = edge_points_outer // (2*vc_max)

        edge_points_next = edge_points_outer
        triangle_filled = []

        i = 1
        for v1_next_x, v1_next_y, v2_next_x, v2_next_y, v3_next_x, v3_next_y in zip(
                np.linspace(v1_outer[0], c[0], it),
                np.linspace(v1_outer[1], c[1], it),
                np.linspace(v2_outer[0], c[0], it),
                np.linspace(v2_outer[1], c[1], it),
                np.linspace(v3_outer[0], c[0], it),
                np.linspace(v3_outer[1], c[1], it)
        ):
            v1_next = (v1_next_x, v1_next_y)
            v2_next = (v2_next_x, v2_next_y)
            v3_next = (v3_next_x, v3_next_y)

            triangle_filled += cls.create_triangle(v1_next, v2_next, v3_next, edge_points=edge_points_next)
            #
            # if i < 0.2*it:
            #     scale_factor = 26
            # elif 0.2*it <= i < 0.4*it:
            #     scale_factor = 24
            # elif 0.4*it <= i < 0.6:
            #     scale_factor = 22
            # if 0.6*it <= i < 0.8*it:
            #     scale_factor = 12
            # else:
            #     scale_factor = 6
            if i < 0.6*it:
                scale_factor = vc_max * np.sqrt(i) * 2 / i
            else:
                scale_factor = vc_max * np.sqrt(i) * 2 / i + 12
            # scale_factor = edge_points_outer / (vc_max * (i + 1))
            edge_points_next -= int(scale_factor)
            i += 1

        return triangle_filled


class ComplexTrig(Complex):
    def __init__(self, phi, r=1, ndigits=3):
        real = r * np.cos(phi)
        imag = r * np.sin(phi)

        super().__init__(real, imag, ndigits)


class Qubit:
    def __init__(self, coef):
        bits_superpos, coef_len = self.__qlen(len(coef))

        self._bits_superpos = bits_superpos
        self._coef_len = coef_len
        self._coef = coef

        if len(coef) < self._coef_len:
            for i in range(self._coef_len - len(coef)):
                self._coef.append(Complex(0, 0))

        self.normalize()

    @property
    def bits_superpos(self):
        return self._bits_superpos

    @property
    def coef_len(self):
        return self._coef_len

    @property
    def coef(self):
        return self._coef

    def __str__(self):
        s = ''

        for i in range(self._coef_len):
            if self._coef[i] != 0:
                if self._coef[i].real_rounded in (1, -1) and self._coef[i].imag_rounded == 0:
                    if self._coef[i].real_rounded == -1 and s.strip() == '':
                        s += f'-|{i:0{self._bits_superpos}b}>'
                    else:
                        s += f'|{i:0{self._bits_superpos}b}>'
                elif self._coef[i].real_rounded > 0 and self._coef[i].imag_rounded == 0:
                    s += f'{self._coef[i]}|{i:0{self._bits_superpos}b}>'
                elif self._coef[i].real_rounded < 0 and self._coef[i].imag_rounded == 0:
                    if s.strip() == '':
                        s += f'{self._coef[i]}|{i:0{self._bits_superpos}b}>'
                    else:
                        s += f'{-self._coef[i]}|{i:0{self._bits_superpos}b}>'
                elif self._coef[i].real_rounded == 0 and self._coef[i].imag_rounded in (1, -1):
                    if self._coef[i].imag_rounded == -1 and s.strip() == '':
                        s += f'-i|{i:0{self._bits_superpos}b}>'
                    else:
                        s += f'i|{i:0{self._bits_superpos}b}>'
                elif self._coef[i].real_rounded == 0 and self._coef[i].imag_rounded > 0:
                    s += f'{self._coef[i]}|{i:0{self._bits_superpos}b}>'
                elif self._coef[i].real_rounded == 0 and self._coef[i].imag_rounded < 0:
                    if s.strip() == '':
                        s += f'{self._coef[i]}|{i:0{self._bits_superpos}b}>'
                    else:
                        s += f'{-self._coef[i]}|{i:0{self._bits_superpos}b}>'
                else:
                    s += f'({self._coef[i]})|{i:0{self._bits_superpos}b}>'

            if i != self._coef_len - 1 and self._coef[i + 1] != 0 and s.strip() != '':
                if self._coef[i + 1].real_rounded > 0 and self._coef[i + 1].imag_rounded == 0:
                    s += ' + '
                elif self._coef[i + 1].real_rounded < 0 and self._coef[i + 1].imag_rounded == 0:
                    s += ' - '
                elif self._coef[i + 1].real_rounded == 0 and self._coef[i + 1].imag_rounded > 0:
                    s += ' + '
                elif self._coef[i + 1].real_rounded == 0 and self._coef[i + 1].imag_rounded < 0:
                    s += ' - '
                else:
                    s += ' + '

        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self._coef_len

    def __getitem__(self, index):
        return self._coef[index]

    def index_superpos(self, index):
        if index < 0 or index > self._coef_len - 1:
            raise IndexError('Index out of range')

        return f'{index:0{self._bits_superpos}b}'

    def sum(self):
        return sum(list(map(lambda c: c.modulus() ** 2, self._coef)))

    def normalize(self):
        norm = np.sqrt(sum(list(map(lambda c: c.modulus() ** 2, self._coef))))
        self._coef = list(map(lambda c: c / norm, self._coef))

    def measure(self, modify=True):
        new_coef = []
        index = 0

        random_value = random.random()
        current_sum = 0
        for i in range(self._coef_len):
            current_sum += self._coef[i].modulus() ** 2

            if current_sum > random_value:
                new_coef = [Complex(0, 0) for j in range(i)] + [Complex(1, 0)] + \
                           [Complex(0, 0) for j in range(i + 1, self._coef_len)]
                index = i
                break

        if modify:
            self._coef = new_coef
            return self, index
        else:
            return Qubit(new_coef), index

    def sample(self, size=1000):
        outcome = {}
        for i in range(self._coef_len):
            outcome[self.index_superpos(i)] = 0

        for i in range(size):
            _, index = self.measure(modify=False)
            index_superpos = self.index_superpos(index)
            outcome[index_superpos] += 1

        return outcome

    @classmethod
    def print_sample(self, outcome_dict):
        for k, v in outcome_dict.items():
            print(f'{k}: {v}')

    def tensproduct(self, q):
        new_coef = []

        for i in range(self._coef_len):
            for j in range(len(q)):
                new_coef.append(self._coef[i] * q.coef[j])

        return Qubit(new_coef)

    def __qlen(self, list_len):
        n = 0
        is_deg2 = True

        while list_len != 0:
            if list_len != 1 and list_len % 2 != 0:
                is_deg2 = False

            list_len //= 2
            n += 1

        if not is_deg2:
            return n, 2 ** n
        else:
            return n - 1, 2 ** (n - 1)

    @classmethod
    def random(cls, coef_size=2):
        return Qubit(Complex.randlist(coef_size))

    @classmethod
    def randlist(cls, list_size=20, qubit_size=2):
        return [Qubit.random(qubit_size) for i in range(list_size)]

    @classmethod
    def plot(cls, qubit, color_string='bo', label='', markersize=None):
        if isinstance(qubit, Qubit):
            Complex.plot(qubit.coef, color_string, label=label, markersize=markersize)
        elif type(qubit) == list:
            Complex.plot(qubit, color_string, label=label, markersize=markersize)
        else:
            raise TypeError('plot accepts Qubit or list type')


class QubitBloch(Qubit):
    _bloch_sphere = []
    _sphere_created = False
    _init_label = False

    def __init__(self, theta=0, phi=0, a=1, b=1, c=1, coef=None):
        self._a = a
        self._b = b
        self._c = c

        if coef is not None:
            if type(coef) == list or type(coef) == np.ndarray:
                if len(coef) < 2:
                    raise ValueError('incorrect coef length for qubit on Bloch\'s sphere (expected 2)')
                else:
                    self._theta = 2 * np.arccos(coef[0].real)
                    self._phi = coef[1].arg()

                    self._bits_superpos = 1
                    self._coef_len = 2
                    self._coef = coef[:2]
            else:
                raise ValueError('expected list of Complex coef, given {}'.format(type(coef)))
        else:
            self._theta = theta
            self._phi = phi
            self._bits_superpos = 1
            self._coef_len = 2

            coef = [Complex(np.cos(theta / 2)), ComplexTrig(phi) * np.sin(theta / 2)]
            self._coef = coef

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

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @classmethod
    def init_sphere(cls, points=100):
        cls._bloch_sphere = cls.create_sphere(points)
        cls._sphere_created = True

    def show(self, ax, s=3, color='r', marker='^', label='qubit on Bloch\'s sphere',
             sphere_s=2, sphere_color='b', sphere_marker='o', sphere_alpha=1.0):
        if not QubitBloch._sphere_created:
            QubitBloch._sphere_created = QubitBloch.create_sphere()

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')

        QubitBloch.qscatter(QubitBloch._bloch_sphere, ax, s=sphere_s, color=sphere_color,
                            marker=sphere_marker, alpha=sphere_alpha, label='Bloch\'s sphere')

        if not QubitBloch._init_label:
            QubitBloch.qscatter([self], ax, s=s, color=color, marker=marker, label=label)
            plt.legend(loc='upper right', fontsize='small')
        else:
            QubitBloch.qscatter([self], ax, s=s, color=color, marker=marker)

        QubitBloch._init_label = True

    @classmethod
    def random(cls):
        theta = np.pi * random.random()
        phi = 2 * np.pi * random.random()

        return QubitBloch(theta, phi)

    @classmethod
    def randlist(cls, list_size=50):
        return [QubitBloch.random() for i in range(list_size)]

    @classmethod
    def create_sphere(cls, points=225):
        sphere = []
        q_root = int(np.sqrt(points))

        for theta in np.linspace(0, np.pi, q_root):
            for phi in np.linspace(0, 2 * np.pi, q_root, endpoint=False):
                sphere.append(QubitBloch(theta, phi))

        if q_root ** 2 < points:
            for i in range(points - q_root ** 2):
                theta = np.pi * random.random()
                phi = 2 * np.pi * random.random()

                sphere.append(QubitBloch(theta, phi))

        return sphere

    @classmethod
    def create_elipsoid(cls, a, b, c, points=225):
        elipsoid = []
        q_root = int(np.floor(np.sqrt(points)))

        for theta in np.linspace(0, np.pi, q_root):
            for phi in np.linspace(0, 2 * np.pi, q_root):
                elipsoid.append(QubitBloch(theta, phi, a=a, b=b, c=c))

        if q_root ** 2 < points:
            for i in range(points - q_root ** 2):
                theta = np.pi * random.random()
                phi = 2 * np.pi * random.random()

                elipsoid.append(QubitBloch(theta, phi, a=a, b=b, c=c))

        return elipsoid

    @classmethod
    def qscatter(cls, qubits, ax, s=2, color='b', marker='o', label='', alpha=1.0):
        if type(qubits) == list or type(qubits) == np.ndarray:
            xs = list(map(lambda q: q.a * np.cos(q.phi) * np.sin(q.theta), qubits))
            ys = list(map(lambda q: q.b * np.sin(q.phi) * np.sin(q.theta), qubits))
            zs = list(map(lambda q: q.c * np.cos(q.theta), qubits))

            ax.scatter(xs, ys, zs, s=s, c=color, marker=marker, label=label, alpha=alpha)

            max_radius = max(qubits[0].a, qubits[0].b, qubits[0].c)
            for axis in 'xyz':
                getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
        else:
            raise TypeError('qscatter accepts list of qubits (class QubitBloch)')


def qft(list_param):
    transformed = []

    if isinstance(list_param, (Qubit, QubitBloch)):
        N = list_param.coef_len

        for k in range(N):
            bk = (1 / np.sqrt(N)) * Complex.csum([list_param.coef[j] * ComplexTrig(2 * np.pi * j * k / N)
                                                  for j in range(N)])

            transformed.append(bk)
    elif type(list_param) == list or type(list_param) == np.ndarray:
        if len(list_param) > 0:
            if isinstance(list_param[0], Complex):
                N = len(list_param)

                for k in range(N):
                    bk = (1 / np.sqrt(N)) * Complex.csum([list_param[j] * ComplexTrig(2 * np.pi * j * k / N)
                                                          for j in range(N)])

                    transformed.append(bk)
            elif isinstance(list_param[0], QubitBloch):
                a, b, c = list_param[0].a, list_param[0].b, list_param[0].c
                transformed = list(map(lambda q: QubitBloch(coef=qft(q.coef), a=a, b=b, c=c), list_param))
            elif isinstance(list_param[0], Qubit):
                transformed = list(map(lambda q: Qubit(qft(q.coef)), list_param))
            else:
                raise TypeError('unsupported object type for QFT')
        else:
            raise ValueError('empty list of coef')
    else:
        raise TypeError('unsupported collection type for QFT')

    return transformed
