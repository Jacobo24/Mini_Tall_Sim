import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros del problema
L = 1.0        # Longitud de la barra (0 a L)
N = 10         # Número de nodos (incluyendo extremos)
alpha = 1.0    # Difusividad térmica
T_left = 100   # Temperatura fija en x=0
T_right = 100  # Temperatura fija en x=L
T_init = 20    # Temperatura inicial en la barra (valor interior)
t_max = 1000   # Tiempo total de simulación (s)

# Parámetros para el intercambio convectivo con el ambiente
T_amb = 20     # Temperatura del ambiente
h = 0.1        # Coeficiente de transferencia de calor reducido para una evolución más lenta

# Cálculo de dx y dt (condición de estabilidad: lambda <= 0.5)
dx = L / (N - 1)
dt = 0.5 * dx**2 / alpha
lam = alpha * dt / dx**2

# Número total de pasos de tiempo
n_steps = int(t_max / dt)

# Eje espacial
x = np.linspace(0, L, N)

# Estado inicial de la temperatura
T = np.ones(N) * T_init
T[0] = T_left
T[-1] = T_right

# -- Configuración de la figura y la animación --
fig, ax = plt.subplots()
ax.set_title(f"Evolución estacionaria de la temperatura (N={N})")
ax.set_xlabel("Posición x (m)")
ax.set_ylabel("Temperatura (°C)")
ax.set_xlim(0, L)
ax.set_ylim(min(T_init, T_amb) - 5, T_left + 5)

(line,) = ax.plot(x, T, marker='o', color='red')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

def update(frame):
    """
    Actualiza la temperatura en cada nodo interior aplicando:
      - La difusión mediante diferencias finitas,
      - El intercambio convectivo con el ambiente (con h reducido).
    Los extremos se mantienen fijos.
    """
    global T
    T_old = T.copy()
    # Actualización para cada nodo interior
    for i in range(1, N-1):
        T[i] = T_old[i] + lam * (T_old[i+1] - 2*T_old[i] + T_old[i-1]) - h * dt * (T_old[i] - T_amb)
    # Reaplicar condiciones de frontera (extremos fijos)
    T[0] = T_left
    T[-1] = T_right
    
    line.set_ydata(T)
    time_text.set_text(f"Tiempo: {frame * dt:.2f} s")
    return (line, time_text)

ani = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True, repeat=False)
plt.show()