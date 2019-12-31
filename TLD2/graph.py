import matplotlib.pyplot as plt
import numpy as np


MOTA = [0, 58, 57, 55, 54, 51, 47, 43, 39, 33, 29]

plt.title("Result")
plt.xlabel("Number Of Objects")
plt.ylabel("MOTA(%)")
plt.plot(MOTA, 'ro')
plt.show()
