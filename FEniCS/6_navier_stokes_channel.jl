#FEniCS tutorial demo program: Incompressible Navier-Stokes equations
#for channel flow (Poisseuille) on the unit square using the
#Incremental Pressure Correction Scheme (IPCS).
#
#  u' + u . nabla(u) - div(sigma(u, p)) = f
   
using FEniCS

T = 1.0             # final time (increase to 10 for full problem)
num_steps = 50      # number of time steps (increase for stability)
dt = T / num_steps  # time step size
mu = 1              # kinematic viscosity
rho = 1             # density

# Create mesh and define function spaces
mesh = UnitSquareMesh(16, 16)
V = VectorFunctionSpace(mesh, "P", 2) # Velocity space
Q = FunctionSpace(mesh, "P", 1)       # Pressure space

# Define boundaries
inflow = "near(x[0], 0)"
outflow = "near(x[0], 1)"
walls = "near(x[1], 0) || near(x[1], 1)"

# Define boundary conditions
bcu_noslip = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow = DirichletBC(Q, Constant(8), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)

bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = FeFunction(V)
u_ = FeFunction(V)
p_n = FeFunction(Q)
p_ = FeFunction(Q)

# Define expressions used in variational forms
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant((0,0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Exact velocity expression (Poiseuille flow)
u_e_expr = Expression(("4*x[1]*(1.0 - x[1])", "0"), degree = 2)

# Define strain-rate tensor
function epsilon(u)
    return sym(nabla_grad(u))
end

# Define stress tensor
function sigma(u, p)
    return 2 * mu * epsilon(u) - p * Identity(len(u))
end

# Define variational problem for step 1 (Tentative velocity step)
F1 = rho * dot((u - u_n) / k, v) * dx + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx + 
    inner(sigma(U, p_n), epsilon(v)) * dx + dot(p_n * n, v) * ds -
    dot(mu * nabla_grad(U) * n, v) * ds - dot(f, v) * dx

a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2 (Pressure update)
a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / k) * div(u_) * q * dx

# Define variational problem for step 3 (Velocity update)
a3 = dot(u, v) * dx
L3 = dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx


# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[apply(bc, A1) for bc in bcu]
[apply(bc, A2) for bc in bcp]

# Create VTK file for saving solution
mkpath("Navier_Stokes_Channel")
vtkfile_u = File("Navier_Stokes_Channel/velocity.pvd")
vtkfile_p = File("Navier_Stokes_Channel/pressure.pvd")

# Time-stepping
global t = 0

for n in 0:(num_steps-1)
    # Update current time
    global t += dt

    # update time in expression
    u_e_expr.pyobject.t = t 

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [apply(bc, b1) for bc in bcu]
    solve(A1, vector(u_), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [apply(bc, b2) for bc in bcp]
    solve(A2, vector(p_), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, vector(u_), b3)

    # Interpolate exact solution
    u_e = interpolate(u_e_expr, V)

    # Compute error
    error_L2 = errornorm(u_e, u_, norm="L2")
    error_max = maximum(abs.(get_array(u_e) - get_array(u_)))

    # Norms
    velocity_max = maximum(abs.(get_array(u_)))
    pressure_max = maximum(abs.(get_array(p_)))

    # Print results
    println("Step $n | Time = $t")
    println("  Velocity max (∞-norm): $velocity_max")
    println("  Pressure max:          $pressure_max")
    println("  L2 error in velocity:  $error_L2")
    println("  Max error in velocity: $error_max")
    println("-------------------------------------------------")


    # Save to file
    vtkfile_u << (u_.pyobject, t)
    vtkfile_p << (p_.pyobject, t)

    # Plot solution
    # FEniCS.Plot(u_)

    # Update previous solution
    assign(u_n, u_)
    assign(p_n, p_)

    #println(n)
    # Hold plot
end
