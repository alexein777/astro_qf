import decimal
import math

qubit_coeffs = [0.5, -0.5, -0.5, 0.5]

n = 2
for i in range(2**n): 
    print(f'({qubit_coeffs[i]})|{i:0{n}b}>')

x = decimal.Decimal(0.23525442)
print(f'Zaokruzeno: {x: {4}.{2}}')

y = round(math.sqrt(2) / 2, 3)
print(y)


# inp = '0x0202020202UL'
# format(int(inp[:-2], 16), 'b')

# x = '3'
# print(format(int(x, 16), 'b'))