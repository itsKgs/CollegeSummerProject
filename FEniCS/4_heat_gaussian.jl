#FEniCS tutorial demo program: Diffusion of a Gaussian hill.
#
#  u'= Laplace(u) + f  in a square domain
#  u = u_D             on the boundary
#  u = u_0             at t = 0
#
#  u_D = f = 0
#
#The initial condition u_0 is chosen as a Gaussian hill.

using FEniCS

T = 2.0  # final time
num_steps = 50 # number of time steps
dt = T / num_steps

# Create mesh and define function space
nx = ny = 30
mesh = RectangleMesh(Point([-2.0, -2.0]), Point([2.0, 2.0]), nx, ny)
V = FunctionSpace(mesh, "P", 1)

# Define boundary condition
bc = DirichletBC(V, Constant(0), "on_boundary")

# Define intial value
u_D = Expression("exp(-a*pow(x[0], 2) - a*pow(x[1], 2))", degree=2, a=5)
u_n = interpolate(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = FeFunction(V)

# Create VTK file for saving solution
mkpath("Heat_Gaussian")
vtkfile = File("Heat_Gaussian/solution.pvd")

global t = 0

for n in 0:(num_steps - 1)
    # Update current time
    global t = t + dt
    u_D.pyobject.t = t # update time in expression

    # Compute solution
    lvsolve(a, L, u, bc)

    # Save to file and plot solution
    vtkfile << (u.pyobject, t)

    # Compute exact solution and error
    u_e = interpolate(u_D, V)
    vv = get_array(u_e)
    ww = get_array(u)
    error = maximum(abs.(vv - ww))
    println("t = $t, max error = $error")

    # Update previous solution
    assign(u_n, u)    
end

