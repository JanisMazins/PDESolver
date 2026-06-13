from src.PDESolver import *
from src.pdeplot import plot_pde
import numpy as np

initial = 2.718281828**(-200*(np.linspace(0, 1, 1001) - 0.5)**2)
initial = list(initial)

PDE = PDESolver1D()

PDE.set_pde(pde_str="d2udt2-d2udx2=0")

PDE.set_boundry(start_boundry=(0, "d"), end_boundry=(0, "d"))
PDE.set_initial_condition(initial_function=initial)
PDE.set_interval(time_interval=(0,1), space_interval=(0,1))

PDE.solve()
plot_pde(PDE, ms_per_frame=10)