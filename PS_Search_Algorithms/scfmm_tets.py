"""
https://pythonhosted.org//scikit-fmm/examples.html
https://stackoverflow.com/questions/28187867/geodesic-distance-transform-in-python
"""

import numpy as np
import pylab as plt
import skfmm

X, Y = np.meshgrid(np.linspace(-1, 1, 201), np.linspace(-1, 1, 201))
phi = -1 * np.ones_like(X)

phi[X > -0.5] = 1
phi[np.logical_and(np.abs(Y) < 0.25, X > -0.75)] = 1
plt.contour(X, Y, phi, [0], linewidths=3, colors='black')


mask = np.logical_and(abs(X) < 0.1, abs(Y) < 0.1)
phi = np.ma.MaskedArray(phi, mask)
d = skfmm.distance(phi, dx=1e-2)
plt.title('Travel time from the boundary with an obstacle')
plt.contour(X, Y, phi, [0], linewidths=3, colors='black')
plt.contour(X, Y, phi.mask, [0], linewidths=3, colors='red')
plt.contour(X, Y, d, 15)
plt.colorbar()
plt.savefig('2d_phi_travel_time_mask.png')
plt.show(block=True)
