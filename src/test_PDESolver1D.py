import unittest
import numpy as np
from PDESolver1D import PDESolver1D
from pdeplot import plot_pde

class TestPDE(unittest.TestCase):

#-----set_boundry-----

    def test_0set_boundry(self):
        """Checks if start boundry is set."""
        PDE = PDESolver1D()
        PDE.set_boundry(start_boundry=(10, "n"))
        self.assertEqual(PDE._start_boundry, (10, "n"))

    def test_1set_boundry(self):
        """Checks if end boundry is set."""
        PDE = PDESolver1D()
        PDE.set_boundry(end_boundry=(10, "n"))
        self.assertEqual(PDE._end_boundry, (10, "n"))

    def test_2set_boundry(self):
        """Checks behavior when no boundry is inputed."""
        PDE = PDESolver1D()
        PDE.set_boundry()
        self.assertEqual(PDE._end_boundry, (0, "d"))   
        self.assertEqual(PDE._start_boundry, (0, "d"))

    def test_3set_boundry(self):
        """Checks behavior when an invalid boundry is inputed."""
        PDE = PDESolver1D()
        self.assertRaises(ValueError, PDE.set_boundry, start_boundry=(None, 1), end_boundry=(None, 1))

    def test_4set_boundry(self):
        """Checks behavior when an expression is passed as the boundry condition tuple's first element."""
        PDE = PDESolver1D()
        strings = ["0",
                   "t",
                   "5*t",
                   "t^2",
                   "2.718281828^t",
                   "t^2-t"]
        answers = [["0"],
                   ["t"],
                   ["5", "*", "t"],
                   ["t", "^", "2"],
                   ["2.718281828", "^", "t"],
                   ["t", "^", "2", "-", "t"]]
        
        for idx, string in enumerate(strings):
            PDE.set_boundry(start_boundry=(string, "d"), end_boundry=(string, "d"))
            self.assertEqual(PDE._end_boundry, (answers[idx], "d"))
            self.assertEqual(PDE._start_boundry, (answers[idx], "d"))

#-----set_step_size-----

    def test_0set_step_size(self):
        """Checks if dx is modified."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0,10))
        PDE.set_step_size(dx=10)
        dx = PDE._dx
        self.assertEqual(dx, 10)

    def test_1set_step_size(self):
        """Checks if dt is modified."""
        PDE = PDESolver1D()
        PDE.set_step_size(dt=10)
        dt = PDE._dt
        self.assertEqual(dt, 10)

    def test_2set_step_size(self):
        """Checks behavior without input kwargs."""
        PDE = PDESolver1D()
        PDE.set_step_size()
        self.assertEqual(PDE._dx, 10**-3)
        self.assertEqual(PDE._dt, 10**-3)

    def test_3set_step_size(self):
        """Checks behavior when an invalid step size is inputed."""
        PDE = PDESolver1D()
        self.assertRaises(TypeError, PDE.set_step_size, dx="1", dt="1")
        
#-----set_interval-----

    def test_0set_interval(self):
        """Checks if space interval is modified."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(10,20))
        self.assertEqual(PDE._space_interval, (10,20))

    def test_1set_interval(self):
        """Checks if time interval is modified."""
        PDE = PDESolver1D()
        PDE.set_interval(time_interval=(10,20))
        self.assertEqual(PDE._time_interval, (10,20))

    def test_2set_interval(self):
        """Check behavior without input kwargs."""
        PDE = PDESolver1D()
        PDE.set_interval()
        self.assertEqual(PDE._space_interval, (0,1))
        self.assertEqual(PDE._time_interval, (0,1))

    def test_3set_interval(self):
        """Checks behavior when an invalid interval is inputed."""
        PDE = PDESolver1D()
        self.assertRaises(ValueError, PDE.set_interval, time_interval=(None, True), space_interval=(None, True))

#-----set_tolerance-----

    def test_0set_tolerance(self):
        """Checks if relative tolerance is modified."""
        PDE = PDESolver1D()
        PDE.set_tolerance(reltol=1)
        self.assertEqual(PDE._reltol, 1)

    def test_1set_tolerance(self):
        """Check behavior without input args."""
        PDE = PDESolver1D()
        self.assertRaises(TypeError, PDE.set_tolerance)

    def test_1set_tolerance(self):
        """Check behavior when an invalid relative tolerance is given."""
        PDE = PDESolver1D()
        self.assertRaises(ValueError, PDE.set_tolerance, reltol=-1)

#-----_match_dx_to_n_x-----

    def test_0_match_dx_to_n_x(self):
        """Tests if the attribute _dx is rounded correctly."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0, 10))
        PDE.set_step_size(dx=11)
        PDE._match_dx_to_n_x()
        self.assertEqual(PDE._dx, 10)

    def test_1_match_dx_to_n_x(self):
        """Tests if a ValueError is raised when incompatible space interval and dx are inputed."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0, 1))
        self.assertRaises(ValueError, PDE.set_step_size, dx=100)

#-----_get_pde_tokens-----

    def test_0_get_pde_tokens(self):
        """Tests if some given PDEs with correct syntax are tokenized correctly."""
        PDE = PDESolver1D()
        strings = ["dudt-d2udx2=0",
                   "d2udt2-d2udx2=0",
                   "d2udt2+d2udx2=0",
                   "u*t*dudx*dudt*d2udx2*d2udt2=1+2+3+4+5+6+7+8+9"]
        answers = [["dudt", "-", "d2udx2", "=", "0"],
                   ["d2udt2", "-", "d2udx2", "=", "0"],
                   ["d2udt2", "+", "d2udx2", "=", "0"],
                   ["u", "*", "t", "*", "dudx", "*", "dudt", "*", "d2udx2", "*", "d2udt2", "=", "1", "+", "2", "+", "3", "+", "4", "+", "5", "+", "6", "+", "7", "+", "8", "+", "9"]]
        
        for i, string in enumerate(strings):
            PDE.set_pde(string)
            self.assertEqual(answers[i], PDE._pde_tokens)

    def test_1_get_pde_tokens(self):
        """Tests if some given PDEs with incorrect syntax are rejected."""
        PDE = PDESolver1D()
        strings = ["+dudt-d2udx2=0",
                   "d2udt22-d2udx2=0",
                   "dd2udt2+d2udx2=0",
                   "uu*t*dudx*dudt*d2udx2*d2udt2*d2udxdt=1+2+3+4+5+6+7+8+9",
                   "dudt-d2udx2",
                   "d2udt2-d2udx2+=1",
                   "+dudt-d2udx2=0",
                   "dudt-d2udx2=0+",
                   "duudt-dudx=0",
                   "tt*dudx=0",
                   "xx*dudx=0",
                   "9.9.*dudx=0"]
        for i, string in enumerate(strings):
            self.assertRaises(ValueError, PDE.set_pde, string)

#-----_get_expression_tokens-----

    def test_0_get_expression_tokens(self):
        """Tests if some given initial conditions with correct syntax are tokenized correctly."""
        PDE = PDESolver1D()
        strings = ["0",
                   "x",
                   "5*x",
                   "x^2",
                   "2.718281828^x",
                   "x^2-x"]
        answers = [["0"],
                   ["x"],
                   ["5", "*", "x"],
                   ["x", "^", "2"],
                   ["2.718281828", "^", "x"],
                   ["x", "^", "2", "-", "x"]]

        for i, string in enumerate(strings):
            PDE.set_initial_condition(initial_function=string, initial_derivative=string)
            self.assertEqual(answers[i], PDE._initial_function)
            self.assertEqual(answers[i], PDE._initial_derivative)

    def test_1_get_expression_tokens(self):
        """Tests if some given initial conditions with incorrect syntax are rejected."""
        PDE = PDESolver1D()
        strings = ["0.00.",
                   "5x",
                   "xx",
                   "x-+2",
                   "f(x)=x",
                   "e^x",
                   "t^2"]

        for i, string in enumerate(strings):
            self.assertRaises(ValueError, PDE.set_initial_condition, string)

#-----set_initial_condition-----

    def test_0_set_initial_condition(self):
        """Tests if incorrectly formatted initial condition lists/tuples are rejected."""
        PDE = PDESolver1D()
        initial_conditions = ([1, 2, "3"],
                              ["1", "2", "3"],
                              [True, False, False],
                              [1, 2, None],
                              {1:1, 2:2, 3:3},
                              1)
        for condition in initial_conditions:
            self.assertRaises(TypeError, PDE.set_initial_condition, initial_function=condition)
            self.assertRaises(TypeError, PDE.set_initial_condition, initial_derivative=condition)

#-----_get_time_order-----

    def test_0_get_time_order(self):
        """Tests if some given PDEs return correct time order."""
        PDE = PDESolver1D()
        strings = {"dudt-d2udx2=0":1,
                   "d2udt2-d2udx2=0":2,
                   "d2udt2+d2udx2=0":2,
                   "u*t*dudx*dudt*d2udx2*d2udt2=1+2+3+4+5+6+7+8+9":2,
                   "d2udx2=0":0,
                   "d2udt2=0":2}

        for string in strings:
            PDE.set_pde(string)
            self.assertEqual(strings[string], PDE._time_order)

#-----_get_initial_condition-----

    def test_0_get_initial_condition(self):
        """Tests if initial function returns expected numpy vector."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0,100))
        PDE.set_step_size(dx=1)
        PDE.set_pde("dudt=0")
        strings = ["0", 
                   "999",
                   "x",
                   "x^2",
                   "2.718281828^x",
                   "x^5-x^4+5*x^3-x^2-x",
                   "1.000001^x*2",
                   "2*x*x*4",
                   "x*x*x*x*x+x",
                   "5*4*3+2*x^2-50"]

        #Tips for new test cases: replace x with linspace(0,100,100) 
        answers = [np.zeros((101,1)),
                   999*np.ones((101,1)),
                   np.linspace(0,100,101),
                   np.linspace(0,100,101)**2,
                   (2.718281828*np.ones((101,1)))**np.linspace(0,100,101).reshape(-1, 1),
                   np.linspace(0,100,101)**5-np.linspace(0,100,101)**4+5*np.linspace(0,100,101)**3-np.linspace(0,100,101)**2-np.linspace(0,100,101),
                   1.000001**np.linspace(0,100,101)*2,
                   8*np.linspace(0,100,101)**2,
                   np.linspace(0,100,101)**5+np.linspace(0,100,101),
                   2*np.linspace(0,100,101)**2+10]

        for i, string in enumerate(strings):
            PDE.set_initial_condition(initial_function=string)
            v0 = PDE._get_initial_condition()
            for j, comp in enumerate(v0):
                ans = abs(comp-answers[i][j])
                self.assertTrue(ans < 0.0005)
            
    def test_1_get_initial_condition(self):
        """Tests if initial derivative returns expected numpy vector. The initial condition for u(x,t) is set to 0."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0,100))
        PDE.set_pde("d2udt2=0")
        PDE.set_step_size(dx=1)
        strings = ["0", 
                   "999",
                   "x",
                   "x^2",
                   "2.718281828^x",
                   "x^5-x^4+5*x^3-x^2-x",
                   "1.000001^x*2",
                   "2*x*x*4",
                   "x*x*x*x*x+x",
                   "5*4*3+2*x^2-50"]

        #Tips for new test cases: replace x with linspace(0,100,100) 
        answers = [np.zeros((101,1)),
                   999*np.ones((101,1)),
                   np.linspace(0,100,101),
                   np.linspace(0,100,101)**2,
                   (2.718281828*np.ones((101,1)))**np.linspace(0,100,101).reshape(-1, 1),
                   np.linspace(0,100,101)**5-np.linspace(0,100,101)**4+5*np.linspace(0,100,101)**3-np.linspace(0,100,101)**2-np.linspace(0,100,101),
                   1.000001**np.linspace(0,100,101)*2,
                   8*np.linspace(0,100,101)**2,
                   np.linspace(0,100,101)**5+np.linspace(0,100,101),
                   2*np.linspace(0,100,101)**2+10]

        for i, string in enumerate(strings):
            PDE.set_initial_condition(initial_function="0", initial_derivative=string)
            v0 = PDE._get_initial_condition()
            n = int(len(v0)/2)
            for j in range(n):
                ans = abs(v0[j,0])
                self.assertTrue(ans < 0.0005)
            for j in range(n, 2*n):
                ans = abs(v0[j,0]-answers[i][j-n])
                self.assertTrue(ans < 0.0005)

    def test_2_get_initial_condition(self):
        """Tests if initial condition functions u(x,0), when inputed as lists or tuples, return a correct initial condition."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(1,100))
        PDE.set_step_size(dx=1)
        PDE.set_pde("dudt=0")

        initial_conditions = ([0 for _ in range(1, 101)],
                              [999 for _ in range(1, 101)],
                              [x for x in range(1, 101)],
                              [x**2 for x in range(1, 101)],
                              [2.718281828**x for x in range(1, 101)],
                              tuple([x**5-x**4+5*x**3-x**2-x for x in range(1, 101)]),
                              tuple([1.000001**x*2 for x in range(1, 101)]),
                              tuple([2*x*x*4 for x in range(1, 101)]),
                              tuple([x*x*x*x*x+x for x in range(1, 101)]),
                              tuple([5*4*3+2*x**2-50 for x in range(1, 101)]))
        
        for i, condition in enumerate(initial_conditions):
            PDE.set_initial_condition(initial_function=condition)
            v0 = PDE._get_initial_condition()
            for j in range(len(v0)):
                ans = abs(v0[j,0]-condition[j])
                self.assertTrue(ans < 0.0005)

    def test_3_get_initial_condition(self):
        """Tests if initial condition derivatives du/dt(x, 0), when inputed as lists or tuples, return a correct initial condition."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0,100))
        PDE.set_pde("d2udt2=0")
        PDE.set_step_size(dx=1)

        initial_conditions = ([0 for _ in range(101)],
                              [999 for _ in range(101)],
                              [x for x in range(0, 101)],
                              [x**2 for x in range(0, 101)],
                              [2.718281828**x for x in range(0, 101)],
                              tuple([x**5-x**4+5*x**3-x**2-x for x in range(0, 101)]),
                              tuple([1.000001**x*2 for x in range(0, 101)]),
                              tuple([2*x*x*4 for x in range(0, 101)]),
                              tuple([x*x*x*x*x+x for x in range(0, 101)]),
                              tuple([5*4*3+2*x**2-50 for x in range(0, 101)]))

        for i, condition in enumerate(initial_conditions):
            PDE.set_initial_condition(initial_function="0", initial_derivative=condition)
            v0 = PDE._get_initial_condition()
            n = int(len(v0)/2)
            for j in range(n):
                ans = abs(v0[j,0])
                self.assertTrue(ans < 0.0005)
            for j in range(n, 2*n):
                ans = abs(v0[j,0]-initial_conditions[i][j-n])
                self.assertTrue(ans < 0.0005)

#-----_is_linear-----

    def test_0_is_linear(self):
        """Checks if function returns correct linearity for a set of given PDEs."""
        derivatives = ("dudx", "dudt", "d2udx2", "d2udt2", "u")
        operations_1 = ("+", "-")
        operations_2 = ("*", "/")
        true_pde_strings = {f"{term_1}{op}{term_2}=0":True for term_1 in derivatives for op in operations_1 for term_2 in derivatives}
        false_pde_strings = {f"{term_1}{op}{term_2}=0":False for term_1 in derivatives for op in operations_2 for term_2 in derivatives}
        strings = true_pde_strings | false_pde_strings
        
        PDE = PDESolver1D()
        for string in strings:
            PDE.set_pde(string)
            self.assertEqual(strings[string], PDE._is_linear(PDE._pde_tokens))

    def test_1_is_linear(self):
        """Checks if function returns correct linearity for a set of more complex PDEs."""
        strings = {"dudt-d2udx2=0":True,
                   "d2udt2-d2udx2=0":True,
                   "d2udt2+d2udx2=0":True,
                   "d2udx2=0":True,
                   "d2udt2=0":True,
                   "u*t+x*t=0":True,
                   "t^2-dudt*t=0":True,
                   "d2udt2*t+dudt*t^2+d2udx2*t^3=0":True,
                   "t*dudt*t*t=0":True,
                   "u^1.0000=dudt":False,
                   "dudt^0-x=0":False,
                   "u*t*dudx*dudt*d2udx2*d2udt2=1+2+3+4+5+6+7+8+9":False,
                   "u*u*dudt=0":False,
                   "d2udt2*x-u*t+d2udx2^2=0":False,
                   "u^2=0":False,
                   "dudt^0.0000-d2udt2^1.000001=0":False,
                   "1/dudt=0":False,
                   "dudt+1/dudx=0":False,
                   "1/dudt^1.000001=0":False}

        PDE = PDESolver1D()
        for string in strings:
            PDE.set_pde(string)
            self.assertEqual(strings[string], PDE._is_linear(PDE._pde_tokens))

#-----_is_time_dependent-----

    def test_0_is_time_dependent(self):
        """Checks if correct matrix time dependency is returned for a set of PDEs."""
        strings = {"dudt-d2udx2=0":False,
                   "d2udt2-d2udx2=0":False,
                   "d2udt2+d2udx2=0":False,
                   "d2udx2=0":False,
                   "d2udt2=0":False,
                   "u*t+x*t=0":True,
                   "u*t*dudx*dudt*d2udx2*d2udt2=1+2+3+4+5+6+7+8+9":True,
                   "t^2-dudt*t=0":True,
                   "d2udt2*t+dudt*t^2+d2udx2*t^3=0":True,
                   "t*dudt*t*t=0":True}

        PDE = PDESolver1D()
        for string in strings:
            PDE.set_pde(string)
            self.assertEqual(strings[string], PDE._A_time_dependent)

    def test_1_is_time_dependent(self):
        """Checks if correct forcing term time dependency is returned for a set of PDEs."""
        strings = {"dudt-d2udx2=0":False,
                   "d2udt2-d2udx2=0":False,
                   "d2udt2+d2udx2=0":False,
                   "d2udx2=0":False,
                   "d2udt2=0":False,
                   "u*t*dudx*dudt*d2udx2*d2udt2=1+2+3+4+5+6+7+8+9":False,
                   "d2udt2*t+dudt*t^2+d2udx2*t^3=0":False,
                   "u*t+x*t=0":True,
                   "t^2-dudt*t=0":True,
                   "t*dudt*t*t+t=0":True,
                   "t*t*t*x*dudx+x*t=0":True,
                   "t=0":True}

        PDE = PDESolver1D()
        for string in strings:
            PDE.set_pde(string)
            self.assertEqual(strings[string], PDE._g_time_dependent)

#-----_get_derivative_coefficient_expressions-----

    def test_0_get_derivative_coefficient_expressions(self):
        """Checks if a set of PDEs get the correct coefficients c1, c2, c3, c4, c5 and r (c1(x,t)*dudx + c2(x,t)*d2udx2 + c3(x,t)*dudt + c4(x,t)*d2udt2 c5(x,t)*u + r(x,t) = 0)."""
        strings = {"dudx=0":(["+","1"], [], [], [], [], ["-","0"]),
                   "d2udx2=0":([], ["+","1"], [], [], [], ["-","0"]),
                   "dudt=0":([], [], ["+","1"], [], [], ["-","0"]),
                   "d2udt2=0":([], [], [], ["+","1"], [], ["-","0"]),
                   "u=0":([], [], [], [], ["+","1"], ["-","0"]),
                   "x=0":([], [], [], [], [], ["+","x","-","0"]),
                   "dudt=d2udx2":([], ["-", "1"], ["+", "1"], [], [], []),
                   "d2udt2*t+dudt*t^2+d2udx2*t^3=x*u*t-x^2":([], ["+","1","*","t","^","3"], ["+","1","*","t","^","2"], ["+","1","*","t"], ["-","x","*","1","*","t"], ["+","x","^","2"])}
        
        PDE = PDESolver1D()
        for string in strings:
            PDE.set_pde(string)
            self.assertEqual(strings[string], PDE._get_derivative_coefficient_expressions())

#-----_evaluate_derivative_coefficient_expressions-----

    def test_0_evaluate_derivative_coefficient_expression(self):
        """Checks if a set of PDEs get the correct numpy vectors for the coefficients c1, c2, c3, c4, c5 and r (c1(x,t)*dudx + c2(x,t)*d2udx2 + c3(x,t)*dudt + c4(x,t)*d2udt2 c5(x,t)*u + r(x,t) = 0)
        evaluated in x and t. Every component of the vectors for c1, ... represent the expression evaluated in x_i."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(1,10))
        PDE.set_step_size(dx=1)
        strings = {"dudx=0":(np.ones((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1))),
                   "d2udx2=0":(np.zeros((10, 1)), np.ones((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1))),
                   "dudt=0":(np.zeros((10, 1)), np.zeros((10, 1)), np.ones((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1))),
                   "d2udt2=0":(np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.ones((10, 1)), np.zeros((10, 1)), np.zeros((10, 1))),
                   "u=0":(np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.ones((10, 1)), np.zeros((10, 1))),
                   "x=0":(np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.linspace(1, 10, 10)),
                   "dudt=d2udx2":(np.zeros((10, 1)), -1*np.ones((10, 1)), np.ones((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1))),
                   "d2udt2*t+dudt*t^2+d2udx2*t^3=x*u*t-x^2":(np.zeros((10, 1)), 8*np.ones((10, 1)), 4*np.ones((10, 1)), 2*np.ones((10, 1)), -2*np.linspace(1, 10, 10), np.linspace(1, 10, 10)**2)}

        expressions = ((["+","1"], [], [], [], [], ["-","0"]),
                       ([], ["+","1"], [], [], [], ["-","0"]),
                       ([], [], ["+","1"], [], [], ["-","0"]),
                       ([], [], [], ["+","1"], [], ["-","0"]),
                       ([], [], [], [], ["+","1"], ["-","0"]),
                       ([], [], [], [], [], ["+","x","-","0"]),
                       ([], ["-", "1"], ["+", "1"], [], [], []),
                       ([], ["+","1","*","t","^","3"], ["+","1","*","t","^","2"], ["+","1","*","t"], ["-","x","*","1","*","t"], ["+","x","^","2"]))

        for idx, string in enumerate(strings):
            PDE.set_pde(string)
            ans = PDE._evaluate_derivative_coefficient_expressions(expressions[idx], t_eval=2)
            for i, arr in enumerate(ans):
                for j, elem in enumerate(arr):
                    self.assertTrue(abs(strings[string][i][j]-arr[j, 0]) < 0.00001)

#-----_get_linear_matrix-----

    def test_0_get_linear_matrix(self):
        """Checks if the returned matrix for A matches the manually calculated theoretical matrix A for a given differential equation."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0,2))
        PDE.set_step_size(dx=1)
        PDE.set_boundry(start_boundry=(3, "n"), end_boundry=(3, "n"))
        PDE.set_pde("dudt+dudx+d2udx2+u+x*t=0")
        expressions = PDE._get_derivative_coefficient_expressions()
        coefficients = PDE._evaluate_derivative_coefficient_expressions(expressions, t_eval=0)
        A = PDE._get_linear_matrix(coefficients)
        answer = np.array([[ 1,  -2,  0],
                           [-0.5, 1, -1.5],
                           [ 0,  -2,  1]])

        for i, row in enumerate(answer):
            for j, col in enumerate(row):
                self.assertEqual(answer[i, j], A[i, j])
    
    def test_1_get_linear_matrix(self):
        """Checks if the returned matrix for A matches the manually calculated theoretical matrix A for a given differential equation."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0,2))
        PDE.set_step_size(dx=1)
        PDE.set_boundry(start_boundry=(3, "n"), end_boundry=(3, "n"))
        PDE.set_pde("d2udt2+dudt+dudx+d2udx2+u+x*t=0")
        expressions = PDE._get_derivative_coefficient_expressions()
        coefficients = PDE._evaluate_derivative_coefficient_expressions(expressions, t_eval=0)
        A = PDE._get_linear_matrix(coefficients)
        answer = np.array([[0,    0,  0,   1,  0,  0],
                           [0,    0,  0,   0,  1,  0],
                           [0,    0,  0,   0,  0,  1],
                           [ 1,  -2,  0,  -1,  0,  0],
                           [-0.5, 1, -1.5, 0, -1,  0],
                           [ 0,  -2,  1,   0,  0, -1]])
        for i, row in enumerate(answer):
            for j, col in enumerate(row):
                self.assertEqual(answer[i, j], A[i, j])

#-----_get_linear_forcing_term-----

    def test_0_get_linear_forcing_term(self):
        """Tests if a correct linear forcing term is produced for a given differential equation of first time order with Neumann boundries."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0,2))
        PDE.set_step_size(dx=1)
        PDE.set_boundry(start_boundry=(7, "n"), end_boundry=(3, "n"))
        PDE.set_pde("dudt+dudx+d2udx2+u+x*t=0")
        expressions = PDE._get_derivative_coefficient_expressions()
        coefficients = PDE._evaluate_derivative_coefficient_expressions(expressions, t_eval=3)
        g = PDE._get_linear_forcing_term(coefficients, t_eval=3)
        answer = np.array([[ 7],
                           [ 3],
                           [-3]])
        for i in range(len(g)):
            self.assertEqual(answer[i, 0], g[i, 0])

    def test_0_get_linear_forcing_term(self):
        """Tests if a correct linear forcing term is produced for a given differential equation of second time order with Neumann boundries."""
        PDE = PDESolver1D()
        PDE.set_interval(space_interval=(0,2))
        PDE.set_step_size(dx=1)
        PDE.set_boundry(start_boundry=(7, "n"), end_boundry=(3, "n"))
        PDE.set_pde("d2udt2+dudt+dudx+d2udx2+u+x*t=0")
        expressions = PDE._get_derivative_coefficient_expressions()
        coefficients = PDE._evaluate_derivative_coefficient_expressions(expressions, t_eval=3)
        g = PDE._get_linear_forcing_term(coefficients, t_eval=3)
        answer = np.array([[ 0],
                           [ 0],
                           [ 0],
                           [ 7],
                           [ 3],
                           [-3]])
        for i in range(len(g)):
            self.assertEqual(answer[i, 0], g[i, 0])

#-----max_val-----

    def test_0max_val(self):
        """Tests if the solution exceeds the analytical maxima of the initial function for the heat equation with dirichlet boundries (heat will dissipate, so initial condition 
        is extremal)."""
        initial_functions = ("x-x^2",
                            "x^3-2*x^2+x",
                            "3*x^3-x^4-3*x^2+x")

        max_vals = (0.26,
                    0.16,
                    0.12)

        PDE = PDESolver1D(pde_str="dudt-d2udx2=0", dx=0.1)
        for idx, func in enumerate(initial_functions):
            PDE.set_initial_condition(initial_function=func)
            PDE.solve()
            self.assertTrue(PDE.max_val() < max_vals[idx])

#-----min_val-----

    def test_0min_val(self):
        """Tests if the solution exceeds the analytical minima of the initial function for the heat equation with dirichlet boundries (heat will dissipate, so initial condition 
        is extremal)."""
        initial_functions = ("x^2-x",
                            "2*x^2-x^3-x",
                            "x^4-3*x^3+3*x^2-x")

        min_vals = (-0.26,
                    -0.16,
                    -0.12)

        PDE = PDESolver1D(pde_str="dudt-d2udx2=0", dx=0.1)
        for idx, func in enumerate(initial_functions):
            PDE.set_initial_condition(initial_function=func)
            PDE.solve()
            self.assertTrue(PDE.min_val() > min_vals[idx])

#-----_separate_into_terms-----
    
    def test_0_separate_into_terms(self):
        """Tests if token lists are separated into terms of given format in answers."""
        tokens = (["0"],
                  ["x"],
                  ["5", "*", "x"],
                  ["x", "^", "2"],
                  ["2.718281828", "^", "x"],
                  ["x", "^", "2", "-", "x"])

        answers = ([["0"]],
                   [["x"]], 
                   [["5", "*", "x"]],
                   [["x", "^", "2"]],
                   [["2.718281828", "^", "x"]],
                   [["x", "^", "2"], ["x"]])

        PDE = PDESolver1D()

        for idx, token in enumerate(tokens):
            ans = PDE._separate_into_terms(token)
            self.assertEqual(ans, answers[idx])

#-----_resize_v0-----

    def test_0_resize_v0(self):
        """Tests if a set of given numpy vectors are resized correctly given two Dirichlet boundries and a PDE with time order 1."""
        before = (np.array([1, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1),
                  np.array([1, 0, 1]).reshape(-1, 1),
                  np.zeros((1000, 1)))
        after = (np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1),
                 np.array([0]).reshape(-1, 1),
                 np.zeros((998, 1)))

        for i, arr in enumerate(before):
            PDE = PDESolver1D(pde_str="dudt-d2udx2=0")
            ans = PDE._resize_v0(arr)
            for j, elem in enumerate(ans):
                self.assertEqual(elem, after[i][j, 0])

    
    def test_1_resize_v0(self):
        """Tests if a set of given numpy vectors are resized correctly given a left Neumann boundry and a right Dirichlet boundry and a PDE with time order 1."""
        before = (np.array([1, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1),
                  np.array([1, 0, 1]).reshape(-1, 1),
                  np.zeros((1000, 1)))
        after = (np.array([1, 0, 0, 0, 0, 0, 0]).reshape(-1, 1),
                 np.array([1, 0]).reshape(-1, 1),
                 np.zeros((999, 1)))

        for i, arr in enumerate(before):
            PDE = PDESolver1D(pde_str="dudt-d2udx2=0", start_boundry=(0, "n"))
            ans = PDE._resize_v0(arr)
            for j, elem in enumerate(ans):
                self.assertEqual(elem, after[i][j, 0])

    def test_2_resize_v0(self):
        """Tests if a set of given numpy vectors are resized correctly given a right Neumann boundry and a left Dirichlet boundry and a PDE with time order 1."""
        before = (np.array([1, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1),
                  np.array([1, 0, 1]).reshape(-1, 1),
                  np.zeros((1000, 1)))
        after = (np.array([0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1),
                 np.array([0, 1]).reshape(-1, 1),
                 np.zeros((999, 1)))

        for i, arr in enumerate(before):
            PDE = PDESolver1D(pde_str="dudt-d2udx2=0", end_boundry=(0, "n"))
            ans = PDE._resize_v0(arr)
            for j, elem in enumerate(ans):
                self.assertEqual(elem, after[i][j, 0])   

    def test_3_resize_v0(self):
        """Tests if a set of given numpy vectors are resized correctly given two Neumann boundries and a PDE with time order 2."""
        before = (np.array([1, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1),
                  np.array([1, 0, 0, 1]).reshape(-1, 1),
                  np.zeros((1000, 1)))
        after = (np.array([1, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1),
                 np.array([1, 0, 0, 1]).reshape(-1, 1),
                 np.zeros((1000, 1)))

        for i, arr in enumerate(before):
            PDE = PDESolver1D(pde_str="dudt-d2udx2=0", start_boundry=(0, "n"), end_boundry=(0, "n"))
            ans = PDE._resize_v0(arr)
            for j, elem in enumerate(ans):
                self.assertEqual(elem, after[i][j, 0])

#-----_evaluate-----

    def test_0_evaluate(self):
        """Tests if a set of expressions token lists are evaluated correctly."""
        expressions = (["1", "+", "2"],
                       ["x", "^", "2"],
                       ["t", "-", "5", "*", "3"],
                       ["7", "/", "x", "^", "x"],
                       ["-", "1", "-", "1"])
        
        var_eval = 4

        answers = (1+2,
                   var_eval**2,
                   var_eval-5*3,
                   7/var_eval**var_eval,
                   -1-1)

        PDE = PDESolver1D()

        for idx, expr in enumerate(expressions):
            ans = PDE._evaluate(expr, var_eval)
            self.assertEqual(ans, answers[idx])

#-----_strip_factor-----

    def test_0_strip_factor(self):
        """Tests if given strings are "stripped" in a set of terms."""
        expressions = (["1", "+", "2"],
                       ["dudx", "*", "2"],
                       ["dudt", "*", "x"],
                       ["7", "/", "x", "*", "d2udx2"],
                       ["d2udt2", "/", "t"])        

        factors = ("dudt",
                   "dudx",
                   "dudt",
                   "d2udx2",
                   "d2udt2")

        answers = (["1", "+", "2"],
                   ["1", "*", "2"],
                   ["1", "*", "x"],
                   ["7", "/", "x", "*", "1"],
                   ["1", "/", "t"])
        
        PDE = PDESolver1D()

        for idx, expr in enumerate(expressions):
            ans = PDE._strip_factor(expr, factors[idx])
            self.assertEqual(answers[idx], ans)

#-----_insert_solution-----

    def test_0_insert_solution(self):
        """Tests if a vector is inserted correctly given Dirichlet boundries."""
        vectors = (np.array([1, 2]).reshape(-1, 1),
                   np.array([1, 2, 3]).reshape(-1, 1))

        answers = (np.array([[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [2, 0, 0, 0],
                             [0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [2, 0, 0, 0, 0],
                             [3, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0,]]))

        t_idx = 0
        t_vals = np.array([0, 1, 2, 3])

        PDE = PDESolver1D()

        for idx, v in enumerate(vectors):
            n = answers[idx].shape[0]
            u_vals = np.zeros((n, n))
            PDE._insert_solution(v, t_idx, t_vals, u_vals)
            for i in range(answers[idx].shape[0]):
                for j in range(answers[idx].shape[1]):
                    self.assertEqual(u_vals[i, j], answers[idx][i, j])



#-----Manual testing-----

    def test_print_test_for_manual_testing(self):
        PDE = PDESolver1D()
        PDE.set_pde("d2udt2-d2udx2=0")
        PDE.set_interval(space_interval=(0,5))
        PDE.set_step_size(dx=1)
        PDE.set_boundry(start_boundry=(6, "n"), end_boundry=(5, "d"))

        expressions = PDE._get_derivative_coefficient_expressions()
        coefficients = PDE._evaluate_derivative_coefficient_expressions(expressions)
        #print(coefficients)
        #print(PDE._get_linear_matrix(coefficients))
        #print(PDE._get_linear_forcing_term(coefficients, t_eval=1))

        return
if __name__ == "__main__":
    unittest.main()