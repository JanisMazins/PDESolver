# PDESolver 
A python library for solving partial differential equations numerically. The purpose of the library is to simplify the process of solving PDEs numerically. The main application of the program is visual simulation of the time evolution of varying differential equations.

## Main features

- PDESolver class with methods to input the properties of your PDE and a solve method for your PDE. 

    - Currently, the types of supported PDEs are 1D linear PDEs of maximal derivative order 2 (no mixed derivatives). 

- plot_pde which automatically plots your PDE:s solution in time using matplotlib.

## Installation
Clone the repository:
<pre>git clone https://github.com/JanisMazins/PDESolver.git </pre>
Then:
```python
import PDESolver as pde
```
Or:
```python
from PDESolver import PDESolver
from PDESolver import plot_pde #Optional; used for plotting
```

### Dependencies
PDESolver uses numpy (ver. 2.2.0) and scipy (ver. 1.17.1) to solve PDEs, so these libraries are required for PDESolver to work. If you intend to use pde_plot, then matplotlib (ver. 3.10.8) is also utilized. Earlier versions probably also work, but I haven't tested it. 
```bash
python3 -m pip install numpy
python3 -m pip install scipy
python3 -m pip install matplotlib #Only if you want to plot :)
```

## What PDEs are we solving?
Currently, the type of PDE which can be solved looks as follows:

Let $u = u(x, t)$ be a function for which there is a linear differential equation of the form:

$\sum_i f_i(x, t)\frac{\partial u}{\partial x} + \sum_i g_i(x, t)\frac{\partial^2 u}{\partial x^2} + \sum_i h_i(x, t)\frac{\partial u}{\partial t} + \sum_i k_i(x, t)\frac{\partial^2 u}{\partial t^2} + \sum_i l_i(x, t)u(x,t) + r(x, t) = 0$

If your PDE is of the above form, then PDESolver will (probably) solve it! 
#### Examples of PDEs with the above form:
- $\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2} = 0$
- $\frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} = 0$
- $5xt\frac{\partial u}{\partial t} - 700\frac{\partial^2 u}{\partial x^2} = x^t$
- $\frac{\partial u}{\partial x} + \frac{\partial^2 u}{\partial x^2} = \frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial t^2}$

