# Methods for PDESolver class

### \_\_init\_\_(\*\*kwargs)

Takes all of the key word arguments described by the set-methods below. For example:
```python
PDE = PDESolver1D(pdestr="dudt-d2udx2=0", initial_condition="x")
```
Here are the equavalent default values for the variables in a PDESolver1D object:
```python
pde_str = None
start_boundry = (0, "d")
end_boundry = (0, "d")
initial_function = "0"
initial_derivative = "0" 
dx = 0.001
dt = 0.001
time_interval = (0, 1)
space_interval = (0, 1)
```

## Set methods

### set_pde(pde_str=None)
- pde_str: 
    - Describes you PDE, with syntax in [inputsyntax.md](inputsyntax.md)
    - Type must be string.
    - Syntax according to [inputsyntax.md](inputsyntax.md).

Example:

```python
PDE.set_pde(pde_str="dudt-d2udx2=0")
```

### set_boundry(start_boundry=None, end_boundry=None)
    
- start_boundry:
    - Represents a condition on the left side of your 1D boundry.
    - Must be tuple or list with two indexes. 
    - First one is either an expression string specified in [inputsyntax.md](inputsyntax.md) or an int or float. The second position must be a string "d" for Dirichlet condition and "n" for Neumann condition.

- end_boundry:
    - Represents a condition on the right side of your 1D boundry.
    - Must be tuple or list with two indexes.
    - First one is either an expression string specified in [inputsyntax.md](inputsyntax.md) or an int or float. The second position must be a string "d" for Dirichlet condition and "n" for Neumann condition.

Examples:

```python
PDE.set_boundry(start_boundry=(10, "d"), end_boundry=("t^2", "n"))
```

### set_initial_condition(initial_function=None, initial_condition=None)

- initial_function:
    - Represents the start your function u(x, 0) at the beginning.
    - Can be written as an expression string according to [inputsyntax.md](inputsyntax.md) or as a list or tuple of you function values for every descrete x-value. 

- initial_derivative:
    - Represents the start your derivative u'(x, 0) at the beginning.
    - Can be written as an expression string according to [inputsyntax.md](inputsyntax.md) or as a list or tuple of you function values for every descrete x-value. 

Examples:

```python
initial = 2.718281828**(-200*(np.linspace(0, 1, 1001) - 0.5)**2)
initial = list(initial)

PDE.set_initial_condition(initial_function=initial, initial_derivative="x^2")
```

### set_step_size(dx=None, dt=None)

- dx:
    - Represents the size intervals between the descrete points in space (the step size). 
    - Must be int or float

- dt:
    - Represents the size intervals between the descrete points in time (the step size). 
    - Must be int or float

Examples:

```python
PDE.set_step_size(dx=10**-2, dt=0.0001)
```

### set_interval(space_interval=None, time_interval=None)

- space_interval:
    - Represents the start and end index on the space axis.
    - Must be a list or tuple of length 2 where each position is either an int or a float. 

- time_interval:
    - Represents the start and end index on the time axis.
    - Must be a list or tuple of length 2 where each position is either an int or a float. 

Examples:

```python
PDE.interval(space_interval=(-1, 1), time_interval=(0, 1))#Solves the PDE for a function u(x, t), -1<x<1 and 0<t<1.
```

### set_relative_tolerance(reltol=None)

- reltol:
    - Represents the tolerance for Newton Raphson when solving non linear systems (not yet implemented). 
    - Must be int or float.

Example:
```python
PDE.set_tolerance(reltol=10**-8)
```

## Main methods

### solve()

- Solves the PDE which has been selected. 
- You must set the PDE with either set_pde() or when you initialize the PDESolver1D object. 
- Returns a tuple (t, x, u) where t and x are numpy 1D arrays and u is a numpy ndarray. 
    - t contains the times at which the function has been evaluated.
    - x contains the discretized positions at which the function u(x, t) has been evaluated. 
    - u contains the function values where each column represents the function values at a certain time and each component in this column represents a position in space.
- If you want to plot the PDE automatically, then refer to plot_pde below.

Example:
```python
t, x, u = PDE.solve()

t[5] #sixth descrete time

x[75] #seventy sixth point in space from the left.

u[5, 3] #same as u(x_6, t_4), given discretized x's and t's
```

### max_val()

- Checks the maximum value of u(x,t). Must have solved the given PDE with the solve method from before. 
- Returns a float. 

Example:
```python
max_val = PDE.max_val() #maximum value of the function. Is float
```

### min_val()

- Checks the minimum value of u(x,t). Must have solved the given PDE with the solve method from before. 
- Returns a float. 

```python
min_val = PDE.min_val() #minimum value of the function.
```

# Methods in pdeplot

### plot_pde(PDE)

- Takes a PDESolver1D object and plots the time evolution of the function with matplotlib.
- You must have solved the PDE before plotting.

Example:
```python
PDE = PDESolver(pde_str="dudt-d2udx2=0")
PDE.solve()
plot_pde(PDE) #Shows time evolution visualization in plot
```