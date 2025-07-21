using FFTW, Plots

# --- 1. Grids ---
x_min, x_max = -5.0, 5.0
N = 256
x = LinRange(x_min, x_max, N)

t_min, t_max = 0.0, π/2
dt = 0.001
Nt = Int(round((t_max - t_min)/dt)) + 1  # include t=0
t = range(t_min, t_max, length=Nt)

# --- 2. Initial condition (complex) ---
ψ = ComplexF64.(2 .* sech.(x))

# --- 3. Fourier wavenumbers & linear propagator ---
L = x_max - x_min
k = vcat(0:N÷2-1, -N÷2:-1) .* (2π / L)
Lprop = exp.(-0.5im .* k.^2 .* dt)  # e^{-i k^2 dt / 2}

# --- 4. Storage ---
ψ_hist = zeros(ComplexF64, Nt, N)
ψ_hist[1, :] .= ψ

# --- 5. Strang split-step loop ---
for n = 2:Nt
    # Nonlinear half-step
    ψ .= ψ .* exp.(-1im .* abs2.(ψ) .* (dt/2))

    # Linear full-step in Fourier space
    ψ̂ = fft(ψ)
    ψ̂ .= ψ̂ .* Lprop
    ψ .= ifft(ψ̂)

    # Nonlinear half-step
    ψ .= ψ .* exp.(-1im .* abs2.(ψ) .* (dt/2))

    # Save
    ψ_hist[n, :] .= ψ
end

# --- 6. Heatmap of |ψ| ---
absψ_hist = abs.(ψ_hist)
heatmap(t, x, absψ_hist'; xlabel="t", ylabel="x", title="|ψ(x,t)|", color=:viridis)

p_top = heatmap(
    t, x, absψ_hist'; 
    xlabel = "t", ylabel = "x",
    colorbar_title = "|h(t,x)|",
    color = :viridis,
    framestyle = :box,
    size = (1000, 350),
    title = "|h(t,x)|"
)

# dashed vertical lines marking window of interest (0.59 and 1.0)
vline!(p_top, [0.59, 1.0]; lc=:black, ls=:dash, lw=1.5, label=false)

# overlay data sampling points (black ×)
scatter!(p_top, t_data, x_data; m=:x, ms=4, c=:black, label="Data ($ndata points)")

using Plots

absψ = abs.(ψ_hist)   # Nt × Nx array (already computed)
Nt, Nx = size(absψ)

anim = @animate for n in 1:Nt
    plot(
        x, absψ[n, :],
        lw = 2,
        xlabel = "x", ylabel = "|h(t,x)|",
        title = "t = $(round(t[n], digits=2))",
        #ylim = (0, maximum(absψ)*1.1),
        legend = true
    )
end

gif(anim, "abs_h_evolution.gif", fps = 10)
