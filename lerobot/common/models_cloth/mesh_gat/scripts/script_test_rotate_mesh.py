import numpy as np

# Define a 90-degree rotation matrix around Z-axis
angle = 90
radius = np.radians(angle)
sn, cs = np.sin(radius), np.cos(radius)
rotate_matrix = np.array([
    [cs, -sn, 0],
    [sn, cs, 0],
    [0, 0, 1]
])
# print(rotate_matrix)

# Define 2D points (assuming Z=0 for simplicity)
mesh = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])  # Two points: (1,0) and (0,1)

# method
method = 'ij,kj->kj'

# Apply rotation
mesh_rotate = np.einsum(method, rotate_matrix, mesh)

np.set_printoptions(formatter={'float': '{:0.1f}'.format})
print(f'method: {method}')
print(f"original mesh: {mesh}")
print(f"rotated mesh: {mesh_rotate}")