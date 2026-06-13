# Rules for inputing PDEs and expressions

When inputing a string for your PDE, initial condition or boundry conditions into the PDESolver() object, you need to use correct syntax to ensure that it is properly interpreted. Here are some examples:
```python
from src.PDESolver import PDESolver1D

PDE = PDESolver(pde_str="dudt-d2udx2=0", initial_function="x", start_boundry=("t^2", "d"), end_boundry=("0", "d"))
```
```python
from src.PDESolver import PDESolver1D

PDE = PDESolver()
PDE.set_pde(pde_str="d2udt2-d2udx2+u=0")
PDE.set_initial_condition(initial_function="x^2-x", initial_derivative="x")
```

Given that you have a function u(x, t) and a differential equation, here is how you would write the different parts:

## Variables, derivatives and numbers
- $u(x, t) \rightarrow$ u
- $x \rightarrow$ x
- $t \rightarrow$ t
- $\frac{\partial u}{\partial x} \rightarrow$ dudx
- $\frac{\partial^2 u}{\partial x^2} \rightarrow$ d2udx2
- $\frac{\partial u}{\partial t} \rightarrow$ dudt
- $\frac{\partial^2 u}{\partial t^2} \rightarrow$ d2udt2
- $floats \rightarrow$ Ex: 9.9999; 141.5152; 1.00000; ...
- $integers \rightarrow$ Ex: 1, 2, 75, 1751, ...

## Operations
- $Addition \rightarrow$ +
- $Subtraction \rightarrow$ -
- $Multiplication \rightarrow$ *
- $Division \rightarrow$ /
- $Exponents \rightarrow$ ^
- $Equals \rightarrow$ =

## Rules for PDE equation:

- There has to be operations between two different, variables, derivatives or numbers, for example:
    - x^2+dudt=0 $-$ ✔
    - xx-d2udt2=0 $-$ ✕
    - 3*x\*dudt=0 $-$ ✔
    - 3x\*dudt=0 $-$ ✕

- There cannot be two operations in a row, and that includes =. Examples:
    - x**2... $-$ ✕
    - x+-1... $-$ ✕
    - x=-2 $-$ ✕
    - x-=2 $-$ ✕

- There cannot be any operations at the start or end of your PDE string:
    - -dudt+d2udx2=0 $-$ ✕
    - *x\*dudt-d2udx2=0 $-$ ✕
    - dudt=d2udx2- $-$ ✕

## Rules for expressions
Expression strings can be used when inputing an initial condition or initial derivative with set_initial_condition() or a boundry condition with set_boundry(). The following rules apply to writing expressions:
- For initial conditions, set with set_initial_condition(), the independent variable of the initial conditions are written as "x". Say you have a function u(x, t) with a PDE:
    - u(x, 0) = x^2 $\Rightarrow$ 
    ```python
    PDE = PDESolver()
    PDE.set_initial_condition(initial_function="x^2")
    ```
    - u(x, 0) = e^x, $\frac{\partial u}{\partial t}$(x, 0) = x^2 $\Rightarrow$ 
    ```python
    PDE = PDESolver()
    PDE.set_initial_condition(initial_function="2.718281828^x", initial_derivative="x^2")
    ```

- For boundry conditions, set with set_boundry_condition(), the independent variable of the is written as "t". Say you have a function u(x, t) with $a<x<b$ with a PDE:
    - u(a, t) = t, u(b, t) = t^2 $\Rightarrow$
    ```python
    PDE = PDESolver()
    PDE.set_boundry(start_boundry=("t", "d"), end=("t^2", "d"))
    ```    
    - $\frac{\partial u}{\partial x}$(a, t) = e^t, u(b, t) = 2*t $\Rightarrow$
    ```python
    PDE = PDESolver()
    PDE.set_boundry(start_boundry=("2.718281828^t", "n"), end=("2*t", "d"))
    ```


- There cannot be two operations in a row. Examples:
    - x**2 $-$ ✕
    - x+-1 $-$ ✕

- There cannot be any operations at the start or end of your expression string:
    - -x^2 $-$ ✕
    - 2*x+ $-$ ✕





## More examples of correct PDE strings
 ```python
    PDE = PDESolver()
    PDE.set_initial_condition(initial_function="x^2*dudt-5*d2udx2=x*t")
    PDE.set_pde
```
