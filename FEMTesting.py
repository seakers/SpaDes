import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def shape(xi):
    x, y, z = tuple(xi)
    N = [1 - x - y - z, x, y, z]
    return np.array(N)

def gradshape(xi):
    dN = [[-1, 1, 0, 0],
          [-1, 0, 1, 0],
          [-1, 0, 0, 1]]
    return np.array(dN)

def getConnections(nodes):
    tri = Delaunay(nodes)
    simplices = tri.simplices

    # def is_degenerate(tetra):
    #     a, b, c, d = nodes[tetra]
    #     mat = np.vstack((b - a, c - a, d - a))
    #     return np.linalg.matrix_rank(mat) < 3

    # for i, tetra in enumerate(simplices):
    #     if is_degenerate(tetra):
    #         nodes[tetra[-1]] += np.random.normal(scale=1e-6, size=3)

    return simplices


###############################
print('create mesh')

l = 1 # length, m
w = 1 # width, m
h = 0.01 # height, m

acc = 9.81 * 6 # m/s^2

E = 73.1e9  # Young's modulus (Pa) for aluminum 2024
nu = 0.33   # Poisson’s ratio for aluminum 2024

rho = 2780  # Density of aluminum (kg/m^3)

tensYield = 324e6  # Yield strength of steel (Pa)

# # Define the number of plates and their dimensions

num_plates = 1
# nx, ny, nz = 11, 11, 3
num_nodes = 200

cornerNodes = np.array([[0, 0, 0], [l, 0, 0], [0, w, 0], [l, w, 0], [0, 0, h], [l, 0, h], [0, w, h], [l, w, h]])
insideNodes = np.random.rand(num_nodes-8, 3) * [l, w, h]
nodes = np.vstack((cornerNodes, insideNodes))
# nodes = np.array([[i * l / (nx - 1), j * w / (ny - 1), k * h / (nz - 1)] for i in range(nx) for j in range(ny) for k in range(nz)])
# print(nodes)

conn = getConnections(nodes)
conn = np.array(conn)

num_nodes = len(nodes)
mesh_lx = l
mesh_ly = w
mesh_lz = h

###############################
print ('material model - 3D')
C = E/(1.0+nu)/(1.0-2.0*nu) * np.array([[1.0-nu,     nu,     nu,     0.0,     0.0,     0.0],
                                        [    nu, 1.0-nu,     nu,     0.0,     0.0,     0.0],
                                        [    nu,     nu, 1.0-nu,     0.0,     0.0,     0.0],
                                        [   0.0,    0.0,    0.0, 0.5-nu,     0.0,     0.0],
                                        [   0.0,    0.0,    0.0,     0.0, 0.5-nu,     0.0],
                                        [   0.0,    0.0,    0.0,     0.0,     0.0, 0.5-nu]])
###############################
print('create global stiffness matrix')
K = np.zeros((3*num_nodes, 3*num_nodes))
gp = np.array([[0.1381966011250105,0.1381966011250105,0.1381966011250105],
               [0.5854101966249685,0.1381966011250105,0.1381966011250105],
               [0.1381966011250105,0.5854101966249685,0.1381966011250105],
               [0.1381966011250105,0.1381966011250105,0.5854101966249685]])
gw = 0.25
B = np.zeros((6, 12))   
for c in conn:
    xIe = nodes[c,:]
    Ke = np.zeros((12, 12))
    for point in gp:
        dN = gradshape(point)
        J  = np.dot(dN, xIe).T
        dN = np.dot(np.linalg.inv(J), dN)
        B[0,0::3] = dN[0,:]
        B[1,1::3] = dN[1,:]
        B[2,2::3] = dN[2,:]
        B[3,0::3] = dN[1,:]
        B[3,1::3] = dN[0,:]
        B[4,1::3] = dN[2,:]
        B[4,2::3] = dN[1,:]
        B[5,0::3] = dN[2,:]
        B[5,2::3] = dN[0,:]
        Ke += np.dot(np.dot(B.T, C), B) * np.abs(np.linalg.det(J)) * gw /24.0

    for i, I in enumerate(c):
        for j, J in enumerate(c):
            K[3*I:3*I+3, 3*J:3*J+3] += Ke[3*i:3*i+3, 3*j:3*j+3]

###############################
print('assign nodal forces and boundary conditions')

f = np.zeros((3*num_nodes))
f[2::3] = -rho * acc * h * w * l / num_nodes

fixed_nodes = np.where(nodes[:, 0] <= 0.01*l)[0]  # Adjust as needed
for node in fixed_nodes:
    dofs = [3*node, 3*node+1, 3*node+2]
    for dof in dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1  # Prevent singularity
        f[dof] = 0       # No displacement at fixed nodes

# heavy_nodes = np.where(nodes[:, 0] >= 0.99*l)[0]  # Adjust as needed
# for node in heavy_nodes:
#     f[3*node+2] += -1e3

# print("Force Vector in z: ",f[2::3])


###############################
print('solving linear system')

min_length = np.min(np.linalg.norm(nodes[conn[:, 1]] - nodes[conn[:, 0]], axis=1))
if min_length < 1e-6:  # Adjust threshold as needed
    print("Warning: Degenerate elements detected!")

# u = np.linalg.solve(K, f)
K_sparse = csr_matrix(K)
u, _ = cg(K_sparse, f, tol=1e-8)
print('Maximum Displacement: ', max(abs(u)), ' m (', max(abs(u))*1000, ' mm)')

###############################
print("Computing Stresses and Von Mises Stress")

stress_node_sum = np.zeros((num_nodes, 6))  # [σ_x, σ_y, σ_z, τ_xy, τ_yz, τ_zx]
stress_node_count = np.zeros(num_nodes)  # Track number of contributions per node

for c in conn:
    xIe = nodes[c, :]
    ue = np.concatenate([u[3*i:3*i+3] for i in c])  # Element displacement vector
    
    for point in gp:
        dN = gradshape(point)
        J = np.dot(dN, xIe).T
        dN = np.dot(np.linalg.inv(J), dN)

        # Compute strain
        B[0,0::3] = dN[0,:]
        B[1,1::3] = dN[1,:]
        B[2,2::3] = dN[2,:]
        B[3,0::3] = dN[1,:]
        B[3,1::3] = dN[0,:]
        B[4,1::3] = dN[2,:]
        B[4,2::3] = dN[1,:]
        B[5,0::3] = dN[2,:]
        B[5,2::3] = dN[0,:]
        
        strain = np.dot(B, ue)
        stress = np.dot(C, strain)  # Hooke's Law

        # Accumulate stress at nodes
        for i, node in enumerate(c):
            stress_node_sum[node] += stress
            stress_node_count[node] += 1

# Average stresses at nodes
stress_nodes = stress_node_sum / stress_node_count[:, None]

# Compute von Mises stress at each node
von_mises_stress = np.sqrt(0.5 * (
    (stress_nodes[:, 0] - stress_nodes[:, 1])**2 +
    (stress_nodes[:, 1] - stress_nodes[:, 2])**2 +
    (stress_nodes[:, 2] - stress_nodes[:, 0])**2 +
    6 * (stress_nodes[:, 3]**2 + stress_nodes[:, 4]**2 + stress_nodes[:, 5]**2)
))

maxVMStress = np.max(von_mises_stress)
print("Max Von Mises Stress:", maxVMStress, " Pa (", maxVMStress/1e6, " MPa)")

if maxVMStress > tensYield:
    print(f"Yield strength exceeded! ({maxVMStress/1e6} MPa > {tensYield/1e6} MPa)")
else:
    print("Yield strength not exceeded.")

###############################
print('plotting displacements')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=u[2::3], cmap='viridis')
plt.colorbar(sc, label='Z Displacement')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
plt.savefig("displacement.png")

print('plotting von mises stress')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=von_mises_stress, cmap='magma')
plt.colorbar(sc, label='Von Mises Stress')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
plt.savefig("VonMises.png")