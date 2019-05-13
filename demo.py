from quantum_fourier import *
import numpy as np

print('~ Complex numbers demo ~\n')
z1 = Complex(-1.5,2)
z2 = Complex(2.2,-1)

print('z1 = {}'.format(z1))
print('z2 = {}'.format(z2))
print()

print('mod(z1) = {}'.format(z1.modulus()))
print('mod(z2) = {}'.format(z2.modulus()))
print()

print('conjugate(z1) = {}'.format(z1.conjugate()))
print('conjugate(z2) = {}'.format(z2.conjugate()))
print()

print('arg(z1) = {}'.format(z1.arg()))
print('arg(z2) = {}'.format(z2.arg()))
print()

print('z1 in trigonometric form: {}'.format(z1.trigform()))
print('z2 in trigonometric form: {}'.format(z2.trigform()))
print()

print('z1 ** 2 = {}'.format(z1 ** 2))
print('z2 ** 0.5 = {}'.format(z2 ** 0.5))
print()

print('-z1 = {} (unary minus)'.format(-z1))
print('z1[0] = z1[-2] = {} (index operator \'[ind]\')'.format(z1[0]))
print('z2[1] = z2[-1] = {} (index operator \'[ind]\')'.format(z2[1]))
print()

print('z1 == z2 -> {}'.format(z1 == z2))
print('z1 != 3 -> {}'.format(z1 != 3))
print()

print('z1 + z2 = {}'.format(z1 + z2))
print('-1 + z2 = {}'.format(-1 + z2))
z1 += z2
print('z1 after z1 += z2: {}'.format(z1))
print()

print('z1 - z2 = {}'.format(z1 - z2))
print('z1 - 2.2 = {}'.format(z1 - 2.2))
z2 -= 5.765
print('z2 after z2 -= 5.765: {}'.format(z2))
print()

print('z1 * z2 = {}'.format(z1 * z2))
print('z1 * 3 = {}'.format(z1 * 3))
z1 *= z2
print('z1 after z1 *= z2: {}'.format(z1))
print()

print('z1 / z2 = {}'.format(z1 / z2))
print('z1 / 2 = {}'.format(z1 / 2))
print('1 / z2 = {}'.format(1 / z2))
z1 /= z2
print('z1 after z1 /= z2: {}'.format(z1))
print()

print('sin(z1) = {}'.format(z1.sin()))
print('cos(z2) = {}'.format(z2.cos()))
print()

print('Complex(r=1, phi=pi/4): z = {}'.format(ComplexTrig(np.pi / 4)))

li = [Complex(1,-2), Complex(4,-5), Complex(3,3)]
print('Complex.csum([{}, {}, {}]) = {} '.format(li[0], li[1], li[2], Complex.csum(li)))
print()

######################
## Qubit operations ##
######################
print('~ Qubits demo ~')
q = Qubit([Complex(1,2), Complex(3,-3), Complex(4,0), Complex(-1, 0), Complex()])
print('q = {}'.format(q))
q.measure()
q.measure()
q.measure()
print()

q1 = Qubit([Complex(1,1), Complex(2,5), Complex(-2,3), Complex(-1, -1)])
q2 = Qubit([Complex(0,1), Complex(3,6), Complex(-1,4), Complex(0.5, 0.2)])
print('q1 = {}'.format(q1))
print('q2 = {}'.format(q2))
print('q1 x q2 = {}'.format(q1.tensproduct(q2)))
print()

q1_Bloch = QubitBloch(np.pi/2, np.pi/4)
q2_Bloch = QubitBloch(np.pi, np.pi/3)
print('QubitBloch(theta=pi/2, phi=pi/4): {}'.format(q1_Bloch))
print('QubitBloch(theta=pi, phi=pi/3): {}'.format(q2_Bloch))

