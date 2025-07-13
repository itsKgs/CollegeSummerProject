#FEniCS tutorial demo program: Nonlinear Poisson equation.
#
#  -div((1+u^2)*grad(u)) = f = −10x−20y−10  in the unit square.
#                   u = u_D = x+2y+1 on the boundary.

using FEniCS

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "P", 1)

u_code = "x[0] + 2 * x[1] + 1"
f_code = "-10 * x[0] - 20 * x[1] - 10"

calc(var) = 1 + var * var

# Define boundary condition
u_D = Expression(u_code, degree=2)
bc = DirichletBC(V, u_D, "on_boundary")

# Define variational problem
u = FeFunction(V)  # Note: not TrialFunction!
v = TestFunction(V)
f = Expression(f_code, degree=2)
F = calc(u) * dot(grad(u), grad(v)) * dx - f * v * dx

# Compute solution
nlvsolve(F, u, bc)

# Compute exact solution and error
u_e = interpolate(u_D, V)
vv = get_array(u_e)
ww = get_array(u)

@show typeof(u_e)
@show typeof(u)
@show vv[1:3]
@show ww[1:3]

error = maximum(abs.(vv - ww))
println("max error = $error")


# Create VTK file for saving solution
mkpath("Poisson_NonLinear")
vtkfile = File("Poisson_NonLinear/solution.pvd")
vtkfile << u.pyobject
vtkfile_u_e = File("Poisson_NonLinear/load.pvd")
vtkfile_u_e << u_e.pyobject
