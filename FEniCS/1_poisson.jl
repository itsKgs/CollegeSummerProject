#FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
#Test problem is chosen to give an exact solution at all nodes of the mesh.
#
#  -Laplace(u) = f    in the unit square
#            u = u_D  on the boundary
#
#  u_D = 1 + x^2 + 2y^2
#    f = -6

using FEniCS

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "P", 1)

# Define boundary condition u_D = 1 + x^2 + 2y^2
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
bc = DirichletBC(V, u_D, "on_boundary")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)

a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Solve the problem
U = FeFunction(V)
lvsolve(a, L, U, bc) #linear variational solver

# Compute error in L2 norm
err = errornorm(u_D, U, norm="L2")
println("L2 error = $err")

# Export solution to file in VTK format
vtkfile = File("poisson/solution.pvd")
vtkfile << U.pyobject

# Optional: Print solution vector
get_array(L) #this returns an array for the stiffness matrix
array_values = get_array(U) #this returns an array for the solution values
println("Solution values at DOFs:\n", array_values)
