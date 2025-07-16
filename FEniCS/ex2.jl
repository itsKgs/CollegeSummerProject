using FEniCS
using PyCall

# Define `Point` from Python dolfin
const Point = pyimport("dolfin").Point

# --- Physical constants ---
k = 1.0
ρ = 1.0
C_p = 505.0
D = 1/(ρ * C_p)
a = 0.01
b = 0.01
c = 0.005
q = 200 * 0.3
v = 0.5

# --- Time stepping ---
dt = 0.05   
time = 1.0  
num_steps = Int(round(time / dt))

# --- Domain and mesh ---
nx, ny, nz = 20, 20, 20
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 0.5), nx, ny, nz)

# --- Function space ---
V = FunctionSpace(mesh, "P", 1)

# --- Boundary condition: T = 300.0 everywhere ---
bc = DirichletBC(V, Constant(300.0), "on_boundary")

# --- Gaussian heat source ---
function Q_expr(t)
    println("Solving at t = $t")

    Q = Expression(
        "A * D * exp(-3*((pow(x[0] - vt,2)/a2) + (pow(x[1] - 0.5,2)/b2) + (pow(x[2],2)/c2)))",
        degree=3,
        A=(6 * sqrt(3) * q) / (a * b * c * π * sqrt(π)), 
        #A = q / (π^(3/2) * a * b * c),
        D = 1 / (ρ * C_p),
        a2=a^2,
        b2=b^2,
        c2=c^2,
        vt=v * t
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
a1 = lhs(F)
L1 = rhs(F)

# --- Time loop + output ---
mkpath("example2")
vtkfile = File("example2/solution.pvd")

global t = 0.0
for step = 1:num_steps
    global t += dt

    Q_t = Q_expr(t)
    L = Q_t * T_test * dx + L1

    #lvsolve(a1, L, T_new, bc)
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
