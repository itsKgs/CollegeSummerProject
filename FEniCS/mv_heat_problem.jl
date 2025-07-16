using FEniCS
using PyCall

# Define `Point` from Python dolfin
#const Point = pyimport("dolfin").Point
const fenics = pyimport("dolfin")  # ← this is required!
const Point = fenics.Point

# --- Physical constants ---
k = 21.5
ρ = 7910.0
C_p = 505.0
D = 1 / (ρ * C_p)
A = 0.3
σ = 13.75e-6
ε = 2e-6
v = 0.8
P = 200.0

# --- Time stepping ---
dt = 7.5e-5   
time = 0.00375       # Final time (s) 3.75ms 
num_steps = Int(round(time / dt)) # 50 step

# --- Domain and mesh ---
Lx = 4.2e-3   # 4200 µm
Ly = 8.4e-4   # 840 µm
Lz = 3.0e-4   # 300 µm

Nx, Ny, Nz = 120, 120, 100  # mesh resolution
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(Lx, Ly, Lz), Nx, Ny, Nz)

# --- Function space ---
V = FunctionSpace(mesh, "P", 1)

# --- Boundary condition: T = 300.0 everywhere ---
#bc = DirichletBC(V, Constant(300.0), "on_boundary")


#surf = "on_boundary && (near(x[2], 3.0e-4) || near(x[0], 0.0) || near(x[0], 4.2e-3) || near(x[1], 0.0) || near(x[1], 8.4e-4))"
#bc = DirichletBC(V, Constant(300.0), surf)

#@pydef mutable struct BCExpr <: fenics.SubDomain
#    function inside(self, x, on_boundary)
#        return on_boundary && (
#            near(x[2], Lz) ||               # z = Lz (bottom)
#            near(x[0], 0.0) || near(x[0], Lx) ||   # x = 0 or x = Lx
#            near(x[1], 0.0) || near(x[1], Ly)      # y = 0 or y = Ly
#        )
#    end
#end

#bc = DirichletBC(V, Constant(300.0), BCExpr())


# --- Gaussian heat source ---
function Q_expr(t)
    println("Solving at t = $t")

    Q = Expression(
        "coef * D * exp(-1*(((pow(x[0]-vt,2) + pow(x[1] - y0,2))/sigma2) + (pow(x[2],2)/eps2)))",
        degree = 2,
        coef = (A * P) / (2 * π * σ^2 * sqrt(2 * π * ε^2)), 
        D = 1 / (ρ * C_p),
        sigma2 = 2 * (σ^2),
        eps2 = 2 * (ε^2),
        vt = v * t,
        y0 = Ly / 2.0
    )
    #println(Q, v*t) #x[0], x[1], x[2], 
    return Q
end

# --- Trial and test functions ---
T_trial = TrialFunction(V)
T_test = TestFunction(V)

# --- Initial condition ---
T_n = interpolate(Constant(300.0), V)

# --- Solution function ---
T = FeFunction(V)

# --- Weak form ---
F = (T_trial - T_n) / dt * T_test * dx + D * k * dot(grad(T_trial), grad(T_test)) * dx
#F = (ρ * C_p) * (T_trial - T_n) / dt * T_test * dx + k * dot(grad(T_trial), grad(T_test)) * dx - Q * T_test * dx
a1 = lhs(F)
L1 = rhs(F)

# --- Time loop + output ---
mkpath("mv_heat_problem")
vtkfile = File("mv_heat_problem/solution.pvd")

global t = 0.0
for step = 1:num_steps
    global t += dt

    Q_t = Q_expr(t)
    L = Q_t * T_test * dx + L1

    lvsolve(a1, L, T, bc, solver_parameters=Dict("linear_solver" => "cg", "preconditioner" => "ilu"))


    # --- Print info ---
    min_T = minimum(get_array(T))
    max_T = maximum(get_array(T))
    println("t = $t, min(T) = $min_T, max(T) = $max_T")
    println("Total heat: ", assemble(T * dx))
    println("-----------------------------------------------------------")

    # Save to file
    vtkfile << (T.pyobject, t)

    # Update solution
    assign(T_n, T)
end
