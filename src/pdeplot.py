import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_pde(PDE, ms_per_frame=10):
    x1 = PDE._space_interval[0]
    x2 = PDE._space_interval[1]
    y1 = PDE.min_val()
    y2 = PDE.max_val()

    t_vals, x_vals, u_vals = PDE.t_vals, PDE.x_vals, PDE.u_vals
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    time_text = ax.text((x2-x1)/10, 1.05*y2, f"t={round(t_vals[0, 0], 3)}")

    def init():
        ax.set_xlim(x1, x2)
        ax.set_ylim(1.15*y1, 1.15*y2)
        return line,

    def update(i):
        y = u_vals[:, i]
        x = x_vals
        line.set_data(x, y)
        time_text.set_text(f"t={round(t_vals[0, i], 3)}")
        return line, time_text

    animation = FuncAnimation(fig, update, frames=t_vals.shape[1], init_func=init, interval=ms_per_frame)
    plt.show()