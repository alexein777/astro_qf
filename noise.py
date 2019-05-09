import matplotlib.pyplot as plt
import math
import random
import numpy as np

def noise(amp=1, amp_scale=1, noise_freq=100):
    """
    Distorts the signal value by maximum amp value scaled to [-amp_scale, amp_scale], 
    interval, with apply frequency of noise_freq.
    """

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

x = np.linspace(0, 2 * np.pi, 300)
noise_gauss = np.random.normal(0, 0.1, 300)
y1 = np.sin(x)
y2 = list(map(lambda c: np.sin(c) + noise(), x))

plt.plot(x, y1 + noise_gauss, color='blue', label="gauss noise")
plt.plot(x, y2, color="red", label="user noise")

plt.legend()
plt.show()

# print(np.sin(5 + 5j))