import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Parameters
dx = 1
dt = 0.01
x_0 = float(input("Enter Initial Value of X , x_0: "))
x_Nx = float(input("Enter Final value of X, x_Nx: "))
t_0 = float(input("Enter Initial Value of t , t_0: "))
t_Nt = float(input("Enter Final value of t, t_Nt: "))

Nt = round((t_Nt - t_0) / dt)
Nx = round((x_Nx - x_0) / dx)

print("Total No. of discrete points of time t:", Nt + 1)
print("Total No. of discrete points of x:", Nx + 1)

k = 1.0
F = (k * dt) / (dx * dx)
print("F =", F)

# Initialize temperature matrix: Nt+1 time steps and Nx+1 space points
T = np.full((Nt + 1, Nx + 1), 27.0)

# Boundary conditions
T[:, 0] = 200.0
T[:, Nx] = 500.0

# Time stepping
for n in range(Nt):
    for i in range(1, Nx):
        T[n + 1][i] = F * T[n][i - 1] + (1 - 2 * F) * T[n][i] + F * T[n][i + 1]

# Output
print("\nTime step results (T[n][i]):")
for n in range(Nt + 1):
    print(f"t = {n * dt:.2f}s: ", end='')
    for i in range(Nx + 1):
        print(f"{T[n][i]:.2f}", end=' ')
    print()


x = np.linspace(x_0, x_Nx, Nx + 1)
time_indices = np.linspace(0, Nt, 6, dtype=int)  # 6 equally spaced time steps

plt.figure(figsize=(10, 6))
for n in time_indices:
    plt.plot(x, T[n], label=f"t = {n * dt:.2f} s")

plt.title("1D Heat Equation (Explicit Scheme)")
plt.xlabel("x")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

# Generate x and t arrays for labeling
x = np.linspace(x_0, x_Nx, Nx + 1)
t = np.linspace(t_0, t_Nt, Nt + 1)

# Plot 2D heatmap
plt.figure(figsize=(12, 6))
plt.imshow(T, aspect='auto', extent=[x_0, x_Nx, t_Nt, t_0], cmap='hot')
plt.colorbar(label="Temperature (°C)")
plt.title("Temperature Distribution over Space and Time")
plt.xlabel("Position x")
plt.ylabel("Time t (s)")
plt.tight_layout()
plt.show()

