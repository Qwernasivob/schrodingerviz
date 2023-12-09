import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import streamlit as st
import os
import base64
from PIL import Image

# Función de onda para un solo estado cuántico
def single_wavefunction(n, L, x, t, hbar, m):
    E_n = n**2 * np.pi**2 * hbar**2 / (2 * m * L**2)
    psi_n_x = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    psi_n_xt = psi_n_x * np.exp(-1j * E_n * t / hbar)
    return psi_n_xt

# Función de onda superpuesta de múltiples estados
def wavefunction(states, L, x, t, hbar, m):
    psi_xt_total = np.zeros_like(x, dtype=complex)
    for n, active in states.items():
        if active:
            psi_xt_total += single_wavefunction(n, L, x, t, hbar, m)
    max_val = np.max(np.abs(psi_xt_total))
    return psi_xt_total / max_val if max_val != 0 else psi_xt_total

# Inicializar estado de Streamlit
if 'running' not in st.session_state:
    st.session_state.running = False
if 'generated' not in st.session_state:
    st.session_state.generated = False

# Crear columnas para los controles y la animación
left_column, right_column = st.columns([1, 2])

# Controles de los estados cuánticos
states = {}
max_n = 0  # Almacenará el valor máximo de n entre los estados activos
with left_column:
    fps = st.slider("FPS", 1, 60, 30)
    for i in range(1, 4):
        n_value = st.slider(f"Número cuántico n{i}", 1, 5, i)
        active = st.checkbox(f"Activar estado n{i}", True)
        states[n_value] = active
        if active:
            max_n = max(max_n, n_value)
    if st.button('Generar Animación'):
        st.session_state.running = True
        st.session_state.generated = False

# Parámetros para la animación
hbar = 1.0545718e-34  # Reduced Planck constant
m = 9.10938356e-31  # Mass of the electron
L = 1e-10  # Longitud de la "caja" en metros
# Ajustar max_t y dt en función del valor máximo de n
if max_n > 0:
    # number of active states
    max_t = 1e-15 / (max_n**2)
    dt = 5e-18 / (max_n**2) / 5
else:
    max_t = 1e-15  # Valor por defecto si no hay estados activos
    dt = 5e-18

# Ruta del archivo GIF
gif_path = 'wavefunction_animation.gif'

# Condición para generar el GIF
if st.session_state.running and not st.session_state.generated:
    fig, ax = plt.subplots(2, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.6)
    x = np.linspace(0, L, 500)
    progress_bar = st.progress(0)

    def init():
        ax[0].set_title("Parte Real e Imaginaria de la Función de Onda")
        ax[0].axhline(0, color='gray')
        ax[0].set_ylim(-1.15, 1.15)
        ax[0].set_xlim(0, L)  # Limitar el eje x a los valores máximos y mínimos
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        # get rid of ticks
        ax[0].tick_params(axis='both', which='both', bottom=False, top=False,
                          labelbottom=False, right=False, left=False, labelleft=False)

        ax[1].set_title("Distribución de Probabilidad")
        ax[1].axhline(0, color='gray')
        ax[1].axvline(0, color='gray')
        ax[1].set_ylim(-1.15, 1.15)
        ax[1].set_xlim(0, L)
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])

    def animate(t):
        ax[0].clear()
        ax[1].clear()
        init()
        psi_xt = wavefunction(states, L, x, t, hbar, m)
        ax[0].plot(x, np.real(psi_xt), color='blue')
        ax[0].plot(x, np.imag(psi_xt), color='red')
        ax[1].plot(x, np.abs(psi_xt)**2, color='green')
        progress_bar.progress(t / max_t)

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, max_t, dt), init_func=init, blit=False)

    # Guardar la animación como GIF usando Pillow
    ani.save(gif_path, writer='pillow', fps=fps, dpi=50)
    st.session_state.generated = True
    progress_bar.empty()

# Mostrar el GIF en la columna derecha
with right_column:
    if st.session_state.generated and os.path.exists(gif_path):
        with open(gif_path, "rb") as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="wavefunction animation">',
                unsafe_allow_html=True,
            )