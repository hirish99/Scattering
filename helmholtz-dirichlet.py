#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Heavily commented Helmholtz exterior Dirichlet BVP with a combined-field IE
#
# Problem (2D acoustics / scalar Helmholtz):
#   Δu + k^2 u = 0 in R^2 \ Ω̄,   u = g on Γ = ∂Ω,   u satisfies Sommerfeld RC at ∞.
#
# We solve for the scattered field u_sc so that (u_inc + u_sc)|_Γ = g.
# For a "sound-soft" obstacle with prescribed Dirichlet g = 0, this is
# simply u_sc|_Γ = -u_inc|_Γ. To avoid resonances, we use the Brakhage–Werner
# combined-field representation and solve:
#   (½ I + D_k - i*η S_k) μ = -u_inc|_Γ
# Then reconstruct u_sc(x) = D_k[μ](x) - i*η S_k[μ](x) in the exterior.
#
# Core ideas shown here:
#  - Boundary (curve) mesh + discretization
#  - QBX setup for high-accuracy layer potentials
#  - Symbolic construction of CFIE operator in pytential
#  - GMRES solve for the boundary density μ
#  - Potential evaluation at targets and a quick plot

import numpy as np
import numpy.linalg as la
import pyopencl as cl

# Array context: wraps a pyopencl queue; required by meshmode/pytential
from meshmode.array_context import PyOpenCLArrayContext

# Mesh generation for a 2D curve (boundary Γ); we'll use a circle
from meshmode.mesh.generation import make_curve_mesh

# Quadrature/discretization on 1D elements (the boundary curve)
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
)

# QBX: Quadrature by Expansion (resolves singular/near-singular integrals)
from pytential.qbx import QBXLayerPotentialSource

# Symbolic layer-potential primitives and kernels
from pytential import sym

# Simple Krylov solver
from pytential.solve import gmres

# (Optional) visualization helpers; you can remove if you don't want plots
import matplotlib.pyplot as plt


# -------------------------------
# 1) Geometry: a circle boundary Γ
# -------------------------------

def circle_param(radius=1.0, center=(0.0, 0.0)):
    """
    Return a parametrization γ: [0, 2π] -> R^2 for a circle of given radius/center.
    """
    cx, cy = center

    def gamma(t):
        return np.array([cx + radius*np.cos(t), cy + radius*np.sin(t)])
    return gamma


# -----------------------------------------
# 2) OpenCL / Array context and discretize Γ
# -----------------------------------------

# Create an OpenCL context/queue. On a multi-device system, you can
# pick the platform/device explicitly (e.g., via PYOPENCL_CTX env var).
cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

# Array context: drives evaluation/compilation on device
actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

# Build a piecewise-polynomial curve mesh for Γ using the parametrization.
# nelements controls panel count; bdry_order is the polynomial order on each panel.
nelements = 128
bdry_order = 7
curve = circle_param(radius=1.0)

# Panel breakpoints in parameter space (uniform)
t_nodes = np.linspace(0.0, 2.0*np.pi, nelements+1)

# make_curve_mesh(γ, panel_breaks, order)
mesh = make_curve_mesh(curve, t_nodes, bdry_order)

# Boundary quadrature/discretization:
# InterpolatoryQuadratureSimplexGroupFactory builds a nodal discretization
# with enough quadrature nodes for accurate integration on 1D segments.
bdry_quad_order = bdry_order + 3
density_discr = Discretization(
    actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order)
)

# -----------------------------------
# 3) QBX layer potential infrastructure
# -----------------------------------
# QBX constructs local expansions about off-surface centers to evaluate
# singular/near-singular layer potentials robustly (on Γ and near Γ).

qbx = QBXLayerPotentialSource(
    density_discr,
    fine_order=bdry_quad_order + 4,  # higher-order quadrature for on-surface eval
    qbx_order=3,                     # expansion order (increase for more accuracy)
    fmm_order=10,                    # treecode/FMM order; governs speed/accuracy
)

# Helpful: wrap geometry for evaluation convenience (esp. multiple sources/targets)
geo = sym.GeometryCollection(qbx)

# -----------------------------
# 4) Helmholtz data and kernels
# -----------------------------
k = 10.0  # wavenumber
kernel = sym.HelmholtzKernel(2)  # 2D Helmholtz Green's function

# Incident plane wave u_inc(x) = exp(i k d·x), choose d = (1, 0)
inc_dir = np.array([1.0, 0.0])
x = sym.nodes(geo)  # boundary nodes (symbolic)
u_inc_sym = sym.exp(1j * k * (inc_dir[0]*x[0] + inc_dir[1]*x[1]))

# Dirichlet data on Γ: g = 0 ("sound-soft"). Then RHS is -u_inc|_Γ.
bc_rhs_sym = -u_inc_sym

# ------------------------------------------------
# 5) Build the CFIE (Brakhage–Werner) boundary op
# ------------------------------------------------
# We form: (½ I + D_k - i η S_k) μ = -u_inc|_Γ
# Notes:
#   S_k: single-layer boundary operator (on-surface limit of potential)
#   D_k: double-layer boundary operator (principal value limit)
#   ½ I: jump term for the double layer (exterior limit)
#   η > 0: coupling parameter (often η ~ k)
#
# In pytential's symbolic layer, S(…) and D(…) produce the potential.
# To obtain *boundary operators* (i.e., the traces/limits on Γ), we
# supply 'qbx_forced_limit' to indicate interior/exterior/average limits.
# For the exterior Dirichlet problem, use the '+' (exterior) limit.

sigma = sym.var("sigma")     # unknown boundary density μ
eta = k                       # a common, effective choice

# On-surface operators (exterior limit).
# - Single-layer trace: S^Γ_k[μ]
# - Double-layer trace (with jump): (½ I + D^Γ_k)[μ]
#   In pytential, D(...) with qbx_forced_limit=+1 returns the correct limit;
#   we add 0.5 * μ explicitly to include the jump term on Γ (exterior).
S_on = sym.S(kernel, sigma, qbx_forced_limit=+1)
D_on = sym.D(kernel, sigma, qbx_forced_limit=+1)

# Build the combined-field operator applied to μ:
cfie_op_mu = 0.5 * sigma + D_on - 1j * eta * S_on

# -----------------------------
# 6) Discretize and solve on Γ
# -----------------------------
# Turn symbolic expressions into callable operators/vectors, then solve
# the linear system for μ using GMRES.

# Discretize RHS: -u_inc|_Γ
rhs = geo.scatter_flattened(
    actx,
    sym.interp(geo, bc_rhs_sym)   # interpolate symbolic boundary expr to nodes
)

# Discretize the CFIE operator into a matrix-like "callable"
# (Actually executed matrix-free through FMM/QBX)
op = sym.make_operator_builder(geo)(cfie_op_mu)

# Solve op(μ) = rhs with GMRES; returns μ as a DOF array on Γ
gmres_tol = 1.0e-12
gmres_maxit = 200

mu, gmres_info = gmres(
    op.scipy_op(actx, "sigma"),  # linear operator (matrix-free)
    rhs,
    tol=gmres_tol,
    maxiter=gmres_maxit,
)

print(f"GMRES iters: {gmres_info.num_iters},  residual: {gmres_info.residual_norm}")

# ---------------------------------------------
# 7) Reconstruct scattered field u_sc in R^2\Γ̄
# ---------------------------------------------
# Using the same combined-field representation:
#   u_sc(x) = D_k[μ](x) - i η S_k[μ](x)
#
# We'll evaluate this on a simple Cartesian grid of targets for plotting.

# Build target grid (avoid evaluating exactly on Γ)
nx, ny = 200, 200
extent = 2.5
xx = np.linspace(-extent, extent, nx)
yy = np.linspace(-extent, extent, ny)
XX, YY = np.meshgrid(xx, yy, indexing="ij")

targets = np.stack([XX.ravel(), YY.ravel()])  # shape (2, Ntargets)

# Mask out the interior of the obstacle (|x| < radius) to plot only exterior field
radius = 1.0
mask_ext = (targets[0]**2 + targets[1]**2) >= (radius + 3e-3)**2

targets_ext = targets[:, mask_ext]

# Build symbolic potentials at targets
u_sc_sym = (
    sym.D(kernel, sigma, target=targets_ext)   # off-surface double-layer potential
    - 1j * eta * sym.S(kernel, sigma, target=targets_ext)
)

# Bind once with 'sigma' -> solution DOFs
bound = sym.bind(geo, u_sc_sym)

# Evaluate scattered field at targets
u_sc = bound(
    actx,
    sigma=mu
)

# Total field u = u_inc + u_sc
# Evaluate u_inc at the *same* target points:
u_inc_targets = np.exp(1j * k * (inc_dir[0]*targets_ext[0] + inc_dir[1]*targets_ext[1]))
u_tot = u_inc_targets + u_sc.get()

# Put results back on the full grid (NaN interior)
U = np.full(XX.size, np.nan, dtype=np.complex128)
U[mask_ext] = u_tot

U = U.reshape((nx, ny))

# -----------------------------
# 8) Simple visualization (|u|)
# -----------------------------
plt.figure()
plt.imshow(np.abs(U).T, origin="lower",
           extent=[-extent, extent, -extent, extent])
plt.colorbar(label="|u_total|")
circle = plt.Circle((0, 0), radius=radius, fill=False, lw=2)
plt.gca().add_patch(circle)
plt.title(f"2D Helmholtz total field magnitude (k={k})")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
