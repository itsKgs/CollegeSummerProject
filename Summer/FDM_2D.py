import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Input parameters
x_0 = float(input("Enter Initial Value of X, x_0: "))
x_Nx = float(input("Enter Final value of X, x_Nx: "))
y_0 = float(input("Enter Initial Value of Y, y_0: "))
y_Ny = float(input("Enter Final value of Y, y_Ny: "))
t_0 = float(input("Enter Initial Value of t, t_0: "))
t_Nt = float(input("Enter Final value of t, t_Nt: "))

dx = 1.0
dy = 1.0
dt = 0.01
k = 1.0

# Calculate discrete points
Nt = int(round((t_Nt - t_0) / dt))
Nx = int((x_Nx - x_0) / dx)
Ny = int((y_Ny - y_0) / dy)

print(f"Total number of discrete time points: {Nt + 1}")
print(f"Total number of discrete x points: {Nx + 1}")
print(f"Total number of discrete y points: {Ny + 1}")

F_x = (k * dt) / (dx * dx)
F_y = (k * dt) / (dy * dy)

print(f"F_x = {F_x}")
print(f"F_y = {F_y}")

# Initialize temperature array
T = np.full((Nt + 1, Nx + 1, Ny + 1), 27.0)

# Apply boundary conditions
for n in range(Nt + 1):
    T[n, :, 0] = 200.0        # y = y_0
    T[n, :, Ny] = 500.0       # y = y_Ny
    T[n, 0, :] = 200.0        # x = x_0
    T[n, Nx, :] = 500.0       # x = x_Nx

# Time evolution using explicit method
for n in range(Nt):
    for i in range(1, Nx):
        for j in range(1, Ny):
            T[n + 1, i, j] = (
                F_x * (T[n, i + 1, j] + T[n, i - 1, j]) +
                F_y * (T[n, i, j + 1] + T[n, i, j - 1]) +
                (1 - 2 * F_x - 2 * F_y) * T[n, i, j]
            )

# Display final temperature distribution
print("\nFinal Temperature Distribution:")
for n in range(Nt + 1):
    print(f"t = {n * dt:.2f} s:")
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            print(f"{T[n, i, j]:.2f}", end="\t")
        print()
    print()

# Create x, y, and t grids
x = np.linspace(x_0, x_Nx, Nx + 1)
y = np.linspace(y_0, y_Ny, Ny + 1)
t = np.linspace(t_0, t_Nt, Nt + 1)

# Plot temperature distribution at selected time steps
time_indices = np.linspace(0, Nt, 6, dtype=int)  # Select 6 time steps

for n in time_indices:
    plt.figure(figsize=(6, 5))
    plt.imshow(T[n], extent=[x_0, x_Nx, y_0, y_Ny], origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Temperature Distribution at t = {n*dt:.2f} s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()


# x, y, t arrays
x = np.linspace(x_0, x_Nx, Nx + 1)
y = np.linspace(y_0, y_Ny, Ny + 1)
X, Y = np.meshgrid(x, y)

# Select time slices to plot
time_indices = np.linspace(0, Nt, 4, dtype=int)  # 4 time slices

for n in time_indices:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # T[n] shape: (Nx+1, Ny+1) → transpose needed for correct orientation
    surf = ax.plot_surface(X, Y, T[n].T, cmap='hot', edgecolor='k', linewidth=0.2)

    ax.set_title(f"3D Temperature Distribution at t = {n * dt:.2f} s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Temperature (°C)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()