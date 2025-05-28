import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(range(10))
axin1 = ax.inset_axes([0.8, 0.1, 0.15, 0.15])
axin2 = ax.inset_axes([5, 7, 1, 1], transform=ax.transData)
# plot a horizen line
axin1.hlines(0.5, 0, 1, color="red")

plt.savefig("tmp.png")
