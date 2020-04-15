import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# example data
X = np.random.randn(1000,)
Y = 0.2 * np.random.randn(1000) + 0.5

h = sns.jointplot(X, Y)

# JointGrid has a convenience function
h.set_axis_labels('x', 'y', fontsize=16)

# or set labels via the axes objects
h.ax_joint.set_xlabel('new x label', fontweight='bold')

# also possible to manipulate the histogram plots this way, e.g.
h.ax_marg_y.grid('on')

# labels appear outside of plot area, so auto-adjust
plt.tight_layout()

plt.show()