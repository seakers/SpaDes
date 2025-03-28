import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import pickle

def shape(xi):
    x, y, z = tuple(xi)
    N = [1 - x - y - z, x, y, z]
    return np.array(N)

def gradshape(xi):
    dN = [[-1, 1, 0, 0],
          [-1, 0, 1, 0],
          [-1, 0, 0, 1]]
    return np.array(dN)

def createPointCloud(structDims, structLocs, structOrients, locations):

    num_panels = len(structDims)
    numComps = len(locations)
    num_nodes = 50
    allNodes = np.zeros((num_panels*num_nodes + numComps, 3))
    # loop through each panel
    for i in range(len(structDims)):
        dims = np.array(structDims[i])
        locs = np.array(structLocs[i])
        orient = structOrients[i]
        # cornerNodes = np.array([[x-l/2, y-w/2, z-h/2], [x+l/2, y-w/2, z-h/2], 
        #                         [x-l/2, y+w/2, z-h/2], [x+l/2, y+w/2, z-h/2], 
        #                         [x-l/2, y-w/2, z+h/2], [x+l/2, y-w/2, z+h/2], 
        #                         [x-l/2, y+w/2, z+h/2], [x+l/2, y+w/2, z+h/2]])
        # insideNodes = (np.random.rand(num_nodes-8, 3)*[l, w, h] + [x-l/2, y-w/2, z-h/2])
        cornerNodes = np.array([[-1,-1,-1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [-1,1,1], [1,-1,1], [1,1,-1], [1,1,1]])
        insideNodes = np.random.rand(num_nodes-8, 3)*2 - 1
        nodes = np.vstack((cornerNodes, insideNodes))
        nodes = nodes*dims/2
        nodes = np.dot(nodes, np.transpose(orient))
        nodes = nodes + locs
        allNodes[i*num_nodes:(i+1)*num_nodes] = nodes
    
    # loop through each component
    for comp in range(numComps):
        allNodes[num_panels*num_nodes + comp] = locations[comp]
    
    return allNodes


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

def structuralAnalysisMain(structDims, structLocs, structOrients, locations, masses):
    ###############################
    # Material model

    num_panels = len(structDims)
    numComps = len(locations)

    acc = 9.81 * 6 # m/s^2

    E = 73.1e9  # Young's modulus (Pa) for aluminum 2024
    nu = 0.33   # Poisson’s ratio for aluminum 2024

    rho = 2780  # Density of aluminum (kg/m^3)

    tensYield = 324e6  # Yield strength of steel (Pa)

    C = E/(1.0+nu)/(1.0-2.0*nu) * np.array([[1.0-nu,     nu,     nu,     0.0,     0.0,     0.0],
                                            [    nu, 1.0-nu,     nu,     0.0,     0.0,     0.0],
                                            [    nu,     nu, 1.0-nu,     0.0,     0.0,     0.0],
                                            [   0.0,    0.0,    0.0, 0.5-nu,     0.0,     0.0],
                                            [   0.0,    0.0,    0.0,     0.0, 0.5-nu,     0.0],
                                            [   0.0,    0.0,    0.0,     0.0,     0.0, 0.5-nu]])

    ###############################
    # Global stiffness matrix

    gp = np.array([[0.1381966011250105,0.1381966011250105,0.1381966011250105],
                [0.5854101966249685,0.1381966011250105,0.1381966011250105],
                [0.1381966011250105,0.5854101966249685,0.1381966011250105],
                [0.1381966011250105,0.1381966011250105,0.5854101966249685]])
    gw = 0.25
    B = np.zeros((6, 12))  

    # make connections here so that they can be redone if matrix is singular
    detJ = 0
    while detJ == 0:
        # print("Creating new nodes")
        nodes = createPointCloud(structDims, structLocs, structOrients, locations)
        num_nodes = len(nodes)

        conn = getConnections(nodes)
        conn = np.array(conn)

        K = np.zeros((3*num_nodes, 3*num_nodes))

        for c in conn:
            xIe = nodes[c,:]
            for point in gp:
                dN = gradshape(point)
                J  = np.dot(dN, xIe).T
                detJ = np.linalg.det(J)
                if detJ == 0:
                    # print("singular :(")
                    break
            if detJ == 0:
                break

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
    # Nodal forces and boundary conditions
    structVol = 0
    for panel in range(num_panels):
        panelVol = structDims[panel][0]*structDims[panel][1]*structDims[panel][2]
        structVol += panelVol

    f = np.zeros((3*num_nodes))
    f[2::3] = -rho * acc * structVol / num_nodes

    for comp in range(numComps):
        node = num_nodes - numComps + comp
        f[3*node+2] += masses[comp]*acc

    # fixed_nodes = np.where(np.abs(nodes[:, 0]) <= 0.03)[0]  # Adjust as needed
    fixed_nodes = np.where((np.sqrt(nodes[:,0]**2 + nodes[:,1]**2) <= 0.95/2) * # for circular fairing of 937 mm with a margin to catch more nodes
                            (nodes[:,2] >= -1.02) * (nodes[:,2] <= -0.98)) # so its only points on the bottom plate
    for node in fixed_nodes:
        dofs = [3*node, 3*node+1, 3*node+2]
        for dof in dofs:
            K[dof, :] = 0
            K[:, dof] = 0
            K[dof, dof] = 1  # Prevent singularity
            f[dof] = 0       # No displacement at fixed nodes

    # print(f[2::3])


    ###############################
    # Solving linear system


    # u = np.linalg.solve(K, f)
    K_sparse = csr_matrix(K)
    u, _ = cg(K_sparse, f, tol=1e-8)
    # print('Maximum Displacement: ', max(np.abs(u)), ' m (', max(np.abs(u))*1000, ' mm)')
    # print(u[2::3])

    ###############################
    # Computing Stresses

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
    stress_nodes = np.where(stress_node_count[:, None] > 0, 
                        stress_node_sum / stress_node_count[:, None], 
                        0)

    # Compute von Mises stress at each node
    von_mises_stress = np.sqrt(0.5 * (
        (stress_nodes[:, 0] - stress_nodes[:, 1])**2 +
        (stress_nodes[:, 1] - stress_nodes[:, 2])**2 +
        (stress_nodes[:, 2] - stress_nodes[:, 0])**2 +
        6 * (stress_nodes[:, 3]**2 + stress_nodes[:, 4]**2 + stress_nodes[:, 5]**2)
    ))

    
    # #######################
    # Plotting, comment out for normal use

    # Plot Displacements
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # sc = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=np.abs(u[2::3]), cmap='viridis')
    # plt.colorbar(sc, label='Z Displacement')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    # plt.savefig("displacement")
    # pickle.dump(fig, open("displacement", "wb"))

    # # Plot Von Mises Stress
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=von_mises_stress, cmap='magma')
    # plt.colorbar(sc, label='Von Mises Stress')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    # plt.savefig("VonMises")
    # pickle.dump(fig, open("VonMises", "wb"))


    maxVMStress = np.max(von_mises_stress)
    # print("Max Von Mises Stress:", maxVMStress, " Pa (", maxVMStress/1e6, " MPa)")

    if maxVMStress > tensYield:
        # print(f"Yield strength exceeded! ({maxVMStress/1e6} MPa > {tensYield/1e6} MPa)")
        return True
    else:
        # print("Yield strength not exceeded.")
        return False
