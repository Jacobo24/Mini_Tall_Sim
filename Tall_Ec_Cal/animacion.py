import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros
L = 1.0        # Longitud de la barra
N = 10         # Número de nodos (incluyendo extremos)
alpha = 1.0    # Difusividad térmica
T_left = 100   # Temperatura en x=0
T_right = 100  # Temperatura en x=L
T_init = 20    # Temperatura inicial en la barra
t_max = 1000   # Tiempo total de simulación (s)

# Cálculo de dx y dt para mantener la estabilidad (lambda <= 0.5)
dx = L / (N - 1)
dt = 0.5 * dx**2 / alpha
lam = alpha * dt / dx**2

# Cantidad total de pasos de tiempo
n_steps = int(t_max / dt)

# Eje espacial
x = np.linspace(0, L, N)

# Vector de temperatura
T = np.ones(N) * T_init
T[0] = T_left
T[-1] = T_right

# -- Configuración de la figura y la animación --
fig, ax = plt.subplots()
ax.set_title(f"Evolución de la temperatura (N={N})")
ax.set_xlabel("Posición x (m)")
ax.set_ylabel("Temperatura (°C)")

# Límites para la gráfica (ajustar según necesidad)
ax.set_xlim(0, L)
ax.set_ylim(T_init - 5, T_right + 5)

# Línea que se actualizará en cada frame
(line,) = ax.plot(x, T, marker='o', color='red')

# Texto para mostrar el tiempo transcurrido
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

def update(frame):
    """
    Esta función se llama en cada frame de la animación.
    Realiza un paso de tiempo en el esquema explícito y
    actualiza la línea con los nuevos valores de T.
    """
    global T
    T_old = T.copy()
    for i in range(1, N-1):
        T[i] = T_old[i] + lam * (T_old[i+1] - 2*T_old[i] + T_old[i-1])
    # Reaplicamos las fronteras (por precaución)
    T[0] = T_left
    T[-1] = T_right
    
    line.set_ydata(T)  # Actualiza los datos de la línea
    
    # Actualizar el texto del tiempo transcurrido
    time_text.set_text(f"Tiempo: {frame * dt:.2f} s")
    
    return (line, time_text)

# Creamos la animación con FuncAnimation
ani = FuncAnimation(
    fig,
    update,
    frames=n_steps,  # Itera en todos los pasos de tiempo
    interval=50,     # Intervalo en ms entre frames (ajusta a tu gusto)
    blit=True,       # Optimización de dibujo
    repeat=False     # Evita que la animación se repita automáticamente
)

# Mostrar la animación
plt.show()