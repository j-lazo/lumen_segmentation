from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


img_dir = '/home/nearlab/Jorge/data/ureteroscopy/data_phantoms/phantom_006/frames_pt01/'
img_name = 'phantom_006_pt01_0080.png'


image = cv2.imread(img_dir + img_name)
image = cv2.resize(image, (300,300))
image = cv2.blur(image, (7, 7))
w, h, d = np.shape(image)

img_ch_1 = image[:, :, 0]/255
img_ch_2 = image[:, :, 1]/255
img_ch_3 = image[:, :, 2]/255

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255




# Make data.
X = np.arange(0, w)
Y = np.arange(0, h)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig = plt.figure()

ax_1 = fig.add_subplot(221, projection='3d')
# Plot the surface.
surf = ax_1.plot_surface(X, Y, img_ch_1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax_1.set_zlim(-.02, 1.01)
ax_1.zaxis.set_major_locator(LinearLocator(10))


ax_2 = fig.add_subplot(222, projection='3d')
# Plot the surface.
surf = ax_2.plot_surface(X, Y, img_ch_2, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax_2.set_zlim(-.02, 1.01)
ax_2.zaxis.set_major_locator(LinearLocator(10))


ax_3 = fig.add_subplot(223, projection='3d')
# Plot the surface.
surf = ax_3.plot_surface(X, Y, img_ch_3, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax_3.set_zlim(-.02, 1.01)
ax_3.zaxis.set_major_locator(LinearLocator(10))


ax_4 = fig.add_subplot(224, projection='3d')
# Plot the surface.
surf = ax_4.plot_surface(X, Y, gray, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax_4.set_zlim(-.02, 1.01)
ax_4.zaxis.set_major_locator(LinearLocator(10))


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()