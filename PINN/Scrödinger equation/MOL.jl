using MethodOfLines, DifferentialEquations, Plots, DomainSets, ModelingToolkit

@parameters t x
@variables h(..) u(..) v(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Domain limits
x_min = -5.0
x_max = 5.0

t_min = 0.0
t_max = π/2


# 2. PDE system
#eqs = [
#   im * Dt(h(t, x)) + 0.5 * Dxx(h(t, x)) + abs(h(t, x))^2  * h(t, x) ~ 0.0,
#   h(t, x) ~ u(t, x) + im * v(t, x)
#]

eqs =[
    Dt(u(t, x)) + 0.5 * Dxx(v(t, x)) + v(t, x) * ((u(t, x))^2 + (v(t, x))^2) ~ 0.0,
    Dt(v(t, x)) - 0.5 * Dxx(u(t, x)) - u(t, x) * ((u(t, x))^2 + (v(t, x))^2) ~ 0.0
]

# 3. Domains
domains = [
    x ∈ Interval(x_min, x_max),
    t ∈ Interval(t_min, t_max)
]

# 4. Initial conditions
h0 = x -> 2 * sech(x)

# 5. Boundary and initial conditions 
#bcs = [
#    h(0, x) ~  2*sech(x),
#    h(t, -5) ~ h(t, 5),
#    Dx(h(t, -5)) ~ Dx(h(t, 5))
#]

bcs = [
    u(0, x) ~ h0(x),
    v(0, x) ~ 0.0,

    u(t, -5) ~ u(t, 5),
    Dx(u(t, -5)) ~ Dx(u(t, 5)),

    v(t, -5) ~ v(t, 5),
    Dx(v(t, -5)) ~ Dx(v(t, 5))
]

@named pdesys = PDESystem(eqs, bcs, domains, [t, x], [u(t, x), v(t, x)])

# 5. Discretize using MethodOfLines (MOL)
N=50
@time discretization = MOLFiniteDifference([x => N], t, approx_order=2)

@time prob = discretize(pdesys, discretization)
@show length(prob.u0) 
@show prob.u0[1:10]          # First 10 initial values
@show prob.tspan

@time sol = solve(prob, Rodas5(); abstol=1e-9, reltol=1e-9, dtmax=1e-3, saveat=0.01)


discrete_x = sol[x]
discrete_t = sol[t]
discrete_u = sol[u(t, x)]
discrete_v = sol[v(t, x)]
discrete_ψ = discrete_u + im * discrete_v

absψ = abs.(discrete_ψ)

anim = @animate for i in eachindex(discrete_t)
    plot(discrete_x, absψ[i, :], label="|ψ(t=$(discrete_t[i]), x)|", xlabel="x", ylabel="|ψ|",
         title="Time: $(discrete_t[i])", legend=:topright)
end
gif(anim, "schrodinger_abs.gif", fps = 2)

τ = 0.79
i = findmin(abs.(discrete_t .- τ))[2]  # nearest saved time index
plot(discrete_x, absψ[i, :];
     xlabel="x", ylabel="|ψ|",
     title="t ≈ $(round(discrete_t[i], digits=3))",
     label="|ψ|")

# Precompute the heatmap matrix
ψ_abs = abs.(discrete_ψ)

# Create a heatmap
heatmap(
    discrete_t, discrete_x, ψ_abs',  # transpose to match (x vs t)
    xlabel = "t", ylabel = "x",
    title = "|ψ(t,x)|",
    colorbar_title = "|ψ|",
    color = :blues,          # similar colormap
    aspect_ratio = :auto
)
