using CUDA
using Plots

# Physical and simulation parameters
L = 1.0                   # Length of the domain
T = 1.0                   # Final time
k = 2.0f0                 # Diffusion coefficient (Float32 for GPU)

N = 64                    # Number of spatial grid points
dx = L / (N - 1)          # Spatial resolution
dt = 0.0001f0             # Time step (stable for explicit Euler)
steps = round(Int, T/dt)


# Initial condition: u(x, 0) = 6 * sin(pi * x)
x = collect(LinRange(0, L, N))
u0_init = 6f0 .* sin.(π .* x)

# Exact solution for comparison
u_exact = (x, t) -> 6f0 * sin.(π * x) * exp(-2f0 * π^2 * t)

# Move to GPU
u = CuArray(u0_init)
u_new = similar(u)

# GPU kernel for heat equation time step
function kernel_heat!(u_new, u, k, dx, dt)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1
    if i >= 2 && i <= length(u) - 1
        u_new[i] = u[i] + dt * k * (u[i-1] - 2f0 * u[i] + u[i+1]) / dx^2
    end
    return
end

# Time stepper function
function heat_step!(u_new, u, k, dx, dt)
    threads = N - 2
    @cuda threads=threads kernel_heat!(u_new, u, k, dx, dt)
end

function heat_step!(u_new, u, k, dx, dt)
    threads = length(u) - 2
    if threads > 0
        @cuda threads=threads kernel_heat!(u_new, u, k, dx, dt)
    end
end

# Bring result to CPU for plotting
u_final = Array(u)

# Plot numerical vs exact solution
plot(x, u_final, label="GPU Numerical", lw=2)
plot!(x, u_exact.(x, T), label="Exact Solution", linestyle=:dash, lw=2)
xlabel!("x")
ylabel!("u(x, T)")
title!("1D Heat Equation on GPU")
