push!(LOAD_PATH,"/home/ysimillides/Downloads/FEniCS.jl-master/src")
using FEniCS
using PyCall
@pyimport fenics
mesh = UnitSquareMesh(8,8)
V = FunctionSpace(mesh,"P",1)
u_D = Expression("1+x[0]*x[0]+2*x[1]*x[1]",degree=2)
bc1 = DirichletBC(V,u_D, "on_boundary")
u=TrialFunction(V)
v=TestFunction(V)
f=Constant(-6.0)
a = dot(grad(u),grad(v))*dx
L = f*v*dx
U = FEniCS.Function(V)
lvsolve(a,L,U,bc1)
errornorm(u_D,U,norm="L2")

saved_sol = File("FEniCS.jl/sol_poisson.pvd")
saved_sol << U.pyobject
