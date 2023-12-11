import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(i):
    plt.cla()
    plt.plot([0, i], [0, i])

fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, interval=50)

plt.show()