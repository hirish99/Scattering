import numpy as np
from pytential import GeometryCollection, bind, sym
from pytential.qbx import QBXLayerPotentialSource
from meshmode.discretization import Discretization
from meshmode.mesh import generate_curve_mesh  # or surface mesh in 3D
from meshmode.dof_array import flatten

# 1. Create boundary geometry + discretization
# Example: unit circle in 2D
mesh = generate_curve_mesh(lambda t: (np.cos(t), np.sin(t)), 
                           np.linspace(0, 2*np.pi, 200), 
                           order=5)
discr = Discretization(mesh, order=5)

# 2. Define QBX source (for singular/near singular quadrature)
src = QBXLayerPotentialSource(discr, fine_order=8, target_order=8,
                              # other QBX parameters like radius, etc.
                              )

# 3. Define symbolic operators
# sym.identity is I; sym.dbl for double-layer
from pytential.symbolic.primitives import Identity, LaplaceDoubleLayer

# Suppose f is boundary data: a function defined on boundary
def f_func(x, y):
    # example: boundary data = x^2 - y^2
    return x**2 - y**2

# Create boundary operator
sigma = sym.var("sigma")
op = 0.5 * Identity(src) + LaplaceDoubleLayer(src)

# Bind to discretization
lhs = bind(src, op).as_matrix()  # this gives a matrix operator
rhs = bind(src, sym.var("f_data")).with_discr(src).eval()

# 4. Solve for sigma
from scipy.sparse.linalg import gmres
sigma_sol, info = gmres(lhs, rhs, tol=1e-9)

# 5. Evaluate u inside domain: set target points
# For example, points in grid inside unit circle
target_points = np.mgrid[-0.9:0.9:40j, -0.9:0.9:40j].reshape(2, -1).T
# Mask to inside circle
mask = np.sum(target_points**2, axis=1) < 1.0
targets = target_points[mask]

# Use layer potential evaluation
u_eval = bind(src, sym.LaplaceDoubleLayer(src).as_potential()).eval(targets, sigma=sigma_sol)

# 6. Compare to exact solution if known
# For example the harmonic extension of f(x,y) = x^2 - y^2 into the disk is u(x,y)=x^2 - y^2
u_exact = targets[:,0]**2 - targets[:,1]**2
error = np.max(np.abs(u_eval - u_exact))

print("Max error:", error)
