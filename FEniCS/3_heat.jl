#FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
#Test problem is chosen to give an exact solution at all nodes of the mesh.
#
#  u'= Laplace(u) + f  in the unit square
#  u = u_D             on the boundary
#  u = u_0             at t = 0
#
#  u = 1 + x^2 + alpha*y^2 + \beta*t
#  f = beta - 2 - 2*alpha

using FEniCS

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, "P", 1)

# Define boundary condition
u_D = Expression("1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t", degree=2, alpha=alpha, beta=beta, t=0)

bc = DirichletBC(V, u_D, "on_boundary")

# Define initial value
u_n = interpolate(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = FeFunction(V)

mkpath("heat")
vtkfile = File("heat/heat_solution.pvd")

global t = 0
for n in 0:(num_steps - 1)
    global t = t + dt
    u_D.pyobject.t = t # update time in expression

    lvsolve(a, L, u, bc)

    vtkfile << (u.pyobject, t)  # save solution at time t

    # Compute exact solution and error
    u_e = interpolate(u_D, V)
    vv = get_array(u_e)
    ww = get_array(u)
    error = maximum(abs.(vv - ww))
    println("t = $t, max error = $error")

    assign(u_n, u) # update u_n for next step
end
