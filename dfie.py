from meshmode.mesh import TensorProductElementGroup
import numpy as np

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureGroupFactory,
)

from pytential import bind, sym
from pytential.target import PointsTarget
from pytools.obj_array import ObjectArray1D, new_1d


# {{{ set some constants for use below

nelements = 20
bdry_quad_order = 4
mesh_order = bdry_quad_order
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 4*bdry_quad_order
fmm_order = 3

# }}}


def main(visualize=False):
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.mesh.generation import generate_sphere
    mesh = generate_sphere(1, 5, uniform_refinement_rounds=1,
                                group_cls=TensorProductElementGroup)

    if visualize:
        from meshmode.mesh.visualization import write_vertex_vtk_file
        write_vertex_vtk_file(mesh, "sphere.vtu")
   
    pre_density_discr = Discretization(
            actx, mesh,
            InterpolatoryQuadratureGroupFactory(bdry_quad_order))
    
    # We are done defining the mesh and the function space. We have chosen SimplexElementGroup for the
    # MeshElementGroup and we have chosen a Nodal Quadrature using Interpolatory Quadrature Group Factory.


    from pytential.qbx import QBXLayerPotentialSource, QBXTargetAssociationFailedError

    # Wraps for QBX, add's QBX specific data structures.
    qbx = QBXLayerPotentialSource(
            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            )

    from sumpy.visualization import FieldPlotter
    fplot = FieldPlotter(np.zeros(3), extent=20, npoints=50)
    targets = actx.from_numpy(fplot.points)

    from pytential import GeometryCollection
    places = GeometryCollection({
        "qbx": qbx,
        "qbx_target_assoc": qbx.copy(target_association_tolerance=0.2),
        "targets": PointsTarget(targets)
        }, auto_where="qbx")
    density_discr = places.get_discretization("qbx")

    # {{{ describe bvp

    from sumpy.kernel import HelmholtzKernel
    kernel = HelmholtzKernel(3)


    #Define DFIE surface densities
    sigma_sym = sym.cse(sym.var("sigma"))
    rho_sym = sym.cse(sym.var("rho"))

    a_sym_vec = sym.cse(sym.make_sym_vector("a", 2))
    b_sym_vec = sym.cse(sym.make_sym_vector("b", 2))

    k = sym.SpatialConstant("k")
    k_0 = sym.SpatialConstant("k_0")
    u = sym.SpatialConstant("u")
    u_0 = sym.SpatialConstant("u_0")
    eps = sym.SpatialConstant("eps")
    eps_0 = sym.SpatialConstant("eps_0")
    omega_sq = k**2/(eps*u)
    omega = k/sym.sqrt(eps*u)

    #Define normal vector, https://documen.tician.de/pytential/symbolic.html
    n_hat = sym.normal(qbx.ambient_dim).as_vector()

    #Define all boundary operators for ease of use later. k is the Helmholtz constant, not kernel.
    def S_vec(k, v_ambient):
        return new_1d(sym.S(kernel, v_ambient, k=k, qbx_forced_limit=+1))
    
    def M_vec(k, v_ambient):
        return new_1d(sym.cross(n_hat, sym.curl(sym.S(kernel, v_ambient, k=k, qbx_forced_limit='avg'))))
    
    def D(k, sigma):
        return sym.D(kernel, sigma, k=k, qbx_forced_limit='avg')
    
    def S(k, sigma):
        return sym.S(kernel, sigma, k=k, qbx_forced_limit=+1)
    
    def Sp(k, sigma):
        return sym.Sp(kernel, sigma, k=k, qbx_forced_limit='avg')
    
    #Unknown vector: a_sym_vec, sigma, b_sym_vec, rho
    #Define first row of eq. 37 in DFIE paper

    a_sym_vec_amb = sym.parametrization_derivative_matrix(3, 2) @ a_sym_vec
    b_sym_vec_amb = sym.parametrization_derivative_matrix(3, 2) @ b_sym_vec

    A11 = (u_0+u)/2 * a_sym_vec_amb + (u_0 * M_vec(k_0, a_sym_vec_amb)- u * M_vec(k, a_sym_vec_amb))
    A12 = -sym.cross(n_hat, u_0*S_vec(k_0, n_hat * sigma_sym)-u*S_vec(k, n_hat * sigma_sym))
    A13 = sym.cross(n_hat, u_0*eps_0*S_vec(k_0, b_sym_vec_amb)- u*eps*S_vec(k, b_sym_vec_amb))
    A14 = sym.grad(3, S(k_0, rho_sym)-S(k, rho_sym))

    #Define second row of eq. 37 in DFIE paper
    A21 = 0
    A22 = (u_0+u)/2 * sigma_sym + (u_0 * D(k_0, sigma_sym) - u * D(k, sigma_sym))
    A23 = sym.div(u_0*eps_0 * S_vec(k_0, b_sym_vec_amb) - u * eps * S_vec(k, b_sym_vec_amb))
    A24 = -omega_sq * (u_0 * eps_0 * S(k_0, rho_sym) - u * eps * S(k, rho_sym))

    #Define third row of eq. 37 in DFIE paper
    A31 = sym.cross(n_hat, sym.curl(sym.curl(S_vec(k_0, a_sym_vec_amb)-S_vec(k, a_sym_vec_amb)))) 
    A32 = sym.cross(n_hat, sym.curl(S_vec(k_0, n_hat*sigma_sym)-S_vec(k, n_hat*sigma_sym)))
    A33 = (eps_0+eps)/2 * b_sym_vec_amb + (eps_0 * M_vec(k_0, b_sym_vec_amb) - eps * M_vec(k, b_sym_vec_amb))
    A34 = 0

    #Define fourth row of eq. 37 in DFIE paper
    A41 = sym.n_dot(sym.curl(eps_0 * u_0 * S_vec(k_0, a_sym_vec_amb) - eps * u * S_vec(k, a_sym_vec_amb)))
    A42 = sym.n_dot((eps_0 * u_0 * S_vec(k_0, n_hat * sigma_sym) - eps * u * S_vec(k, n_hat * sigma_sym)))
    A43 = sym.n_dot((u_0 * eps_0**2 * S_vec(k_0, b_sym_vec_amb) - u * eps**2 * S_vec(k, b_sym_vec_amb)))
    A44 = -(eps_0 + eps)/2 * rho_sym + (eps_0 * Sp(k_0, rho_sym)- eps * Sp(k, rho_sym))

    operator = new_1d([A11 + A12 + A13 + A14,
                       A21 + A22 + A23 + A24,
                       A31 + A32 + A33 + A34,
                       A41 + A42 + A43 + A44])

    bound_op = bind(places, operator)

    # {{{ fix rhs and solve

    r_out = -10 #what does this do? where the incoming wave originates from?
    nodes = actx.thaw(density_discr.nodes())
    source = np.array([r_out, 0, 0], dtype=object)

    k_vec = np.array([1, 0, 0]) #must be normalized, EM wave moves from left to right along x-axis

    def u_incoming_func_E(x): #defines the incoming wave as a function of source, needs to be a vector.
        E_0 = np.array([0, 1, 0])
        dists = x - source
        return E_0 * sym.exp(1j * np.dot(k_vec, dists.as_vector()))
    
    def u_incoming_func_H(x):
        H_0 = np.array([0, 0, 1])/sym.sqrt(u_0/eps_0)
        dists = x - source
        return H_0 * actx.np.exp(1j * np.dot(k_vec, dists.as_vector()))
    
    #Creating rhs from Eq. 39
    x = sym.nodes(3)
    bcf = sym.cross(n_hat, u_incoming_func_E(x)) #how to stick in nodes?
    bcg = sym.cross(-1j * omega * n_hat, u_incoming_func_H(nodes))
    bfq = 0 * nodes
    bcp = sym.cross(-n_hat, eps_0 * u_incoming_func_E(nodes))
    bc = new_1d([bcf, bcg, bfq, bcp])
    
    #bc = u_incoming_func(nodes)
    bvp_rhs = bind(places, sym.var("bc"))(actx, bc=bc)

    1/0

    from pytential.linalg.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(actx, "sigma", dtype=np.float64),
            bvp_rhs, tol=1e-14, progress=True,
            stall_iterations=0,
            hard_failure=True)

    sigma = bind(places, sym.var("sigma"))(
            actx, sigma=gmres_result.solution)

    # }}}

    from meshmode.discretization.visualization import make_visualizer
    bdry_vis = make_visualizer(actx, density_discr, 20)
    bdry_vis.write_vtk_file("laplace.vtu", [
        ("sigma", sigma),
        ])

    # {{{ postprocess/visualize

    repr_kwargs = {
            "source": "qbx_target_assoc",
            "target": "targets",
            "qbx_forced_limit": None}
    representation_sym = (
            sym.S(kernel, inv_sqrt_w_sigma, **repr_kwargs)
            + sym.D(kernel, inv_sqrt_w_sigma, **repr_kwargs))

    try:
        fld_in_vol = actx.to_numpy(
                bind(places, representation_sym)(actx, sigma=sigma))
    except QBXTargetAssociationFailedError as e:
        fplot.write_vtk_file("laplace-dirichlet-3d-failed-targets.vts", [
            ("failed", actx.to_numpy(e.failed_target_flags)),
            ])
        raise

    # fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
    fplot.write_vtk_file("laplace-dirichlet-3d-potential.vts", [
        ("potential", fld_in_vol),
        ])

    # }}}


if __name__ == "__main__":
    main()