#FEniCS tutorial demo program: Deflection of a membrane.
#  -Laplace(w) = p  in the unit circle
#            w = 0  on the boundary
#
#The load p is a Gaussian function centered at (0, 0.6).
using FEniCS

# Create mesh and define function space
domain = Circle(Point([0.0, 0.0]), 1)

mesh = generate_mesh(domain, 64)
V = FunctionSpace(mesh, "P", 2)

# Define boundary condition
w_D = Constant(0)

bc = DirichletBC(V, w_D, "on_boundary")

# Define load
beta = 8
R0 = 0.6
p = Expression("4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))", degree=1, beta=beta, R0=R0)

# Define variational problem 
w = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(w), grad(v)) * dx
L = p*v*dx

# Compute solution
w = FeFunction(V)
lvsolve(a, L, w, bc)

p = interpolate(p, V)

get_array(L)
println(get_array(w))

# Save solution to file in VTK format
mkpath("poisson_membrane")
vtkfile_w = File("poisson_membrane/deflection.pvd")
vtkfile_w << w.pyobject
vtkfile_p = File("poisson_membrane/load.pvd")
vtkfile_p << p.pyobject


