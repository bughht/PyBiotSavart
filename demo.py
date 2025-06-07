import numpy as np
from PyBiotSavart import BiotSavart
from einops import rearrange
import matplotlib.pyplot as plt

# Define coil geometry

# Circular
coil = np.zeros((200, 3))
coil[:, 2] = 0
coil[:, 0], coil[:, 1] = np.cos(np.linspace(0, 8 * np.pi, coil.shape[0], endpoint=False)) * 6e-2, \
    np.sin(np.linspace(0, 8 * np.pi, coil.shape[0], endpoint=False)) * 6e-2

# # Spiral
# coil = np.zeros((200, 3))
# coil[:, 2] = 0
# coil[:, 0], coil[:, 1] = np.linspace(0, 2, coil.shape[0])*np.cos(np.linspace(0, 8*np.pi, coil.shape[0], endpoint=False)) * \
#     6e-2, np.linspace(0, 2, coil.shape[0])*np.sin(
#         np.linspace(0, 8*np.pi, coil.shape[0], endpoint=False))*6e-2

# # Helical
# coil = np.zeros((200, 3))
# coil[:, 2] = np.linspace(-8e-2, 8e-2, coil.shape[0], endpoint=True)
# coil[:, 0], coil[:, 1] = np.cos(np.linspace(0, 8*np.pi, coil.shape[0], endpoint=False)) * \
#     6e-2, np.sin(np.linspace(0, 8*np.pi, coil.shape[0], endpoint=False))*6e-2

# # Variable radius Helical
# coil = np.zeros((200, 3))
# coil[:, 2] = np.linspace(-8e-2, 8e-2, coil.shape[0], endpoint=True)
# coil[:, 0], coil[:, 1] = np.linspace(0, 2, coil.shape[0])*np.cos(np.linspace(0, 8*np.pi, coil.shape[0], endpoint=False)) * \
#     6e-2, np.linspace(0, 2, coil.shape[0])*np.sin(
#         np.linspace(0, 8*np.pi, coil.shape[0], endpoint=False))*6e-2

# # Toroidal
# r0 = 15e-2
# r1 = 4e-2

# coil = np.zeros((400, 3))
# theta = np.linspace(0, 2*np.pi, coil.shape[0])
# alpha = np.linspace(0, 60*np.pi, coil.shape[0])

# coil[:, 2] = r1*np.cos(alpha)
# coil[:, 0] = (r0+r1*np.sin(alpha))*np.cos(theta)
# coil[:, 1] = (r0+r1*np.sin(alpha))*np.sin(theta)


# Define current
current = 1

# Define mesh grid
xmin, xmax, nx = -25e-2, 25e-2, 51
ymin, ymax, ny = -25e-2, 25e-2, 51
zmin, zmax, nz = -25e-2, 25e-2, 51
x = np.linspace(xmin, xmax, nx, endpoint=True)
y = np.linspace(ymin, ymax, ny, endpoint=True)
z = np.linspace(zmin, zmax, nz, endpoint=True)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
mesh_grid = np.stack([xx, yy, zz], axis=-1)
vectorized_grid = rearrange(mesh_grid, "x y z p -> (x y z) p")

# Compute magnetic field
B = BiotSavart(coil, current, vectorized_grid)
mesh_B = rearrange(B, "(x y z) p -> x y z p", x=nx, y=ny, z=nz)

# Visualization
plt.figure(figsize=(10, 4), dpi=200)
ax1 = plt.subplot(1, 2, 1, projection="3d")
ax1.plot3D(coil[:, 0], coil[:, 1], coil[:, 2])
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_zlim(zmin, zmax)
ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(mesh_B[:, 24, :, 2], cmap="bwr")
im2.set_clim(-5e-5, 5e-5)
plt.colorbar(im2, ax=ax2)
plt.tight_layout()

plt.show()
