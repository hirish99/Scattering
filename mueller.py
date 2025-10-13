import numpy as np
import pyopencl as cl
from meshmode.mesh.generation import generate_sphere
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import InterpolatoryQuadratureSimplexGroupFactory

from pytential.qbx import QBXLayerPotentialSource
from meshmode.array_context import PyOpenCLArrayContext

# --- CL / array context ---
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
actx = PyOpenCLArrayContext(queue)  # <- no force_device_scalars

# --- geometry ---
mesh = generate_sphere(1.0, 3)  # <- replacement for generate_icosphere

# --- boundary discretization & QBX ---
bdry_quad_order = 8
pre_density_discr = Discretization(
    actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order)
)

qbx = QBXLayerPotentialSource(
    pre_density_discr,
    fine_order=bdry_quad_order + 4,
    qbx_order=3,
    fmm_order=10,
)
import numpy as np
from pytential.target import PointsTarget
from pytential import bind
import pytential.symbolic.primitives as sym

