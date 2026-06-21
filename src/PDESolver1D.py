import numpy as np
import scipy.sparse as sp
import re
import math
import warnings

class PDESolver1D():
    """Class for solving PDEs using finite difference methods and method of lines."""

    def __init__(self, **kwargs):
        """Initiates all of the attributes of the PDE-object. Can either be passed in with kwargs or entered later with the associated method."""
        self._pde_tokens = None

        self._linear = None
        self._time_order = None
        self._forcing_term = None

        self._A = None
        self._A_time_dependent = None
        self._g = None
        self._g_time_dependent = None
        
        self._start_boundry = (0, "d")
        self._end_boundry = (0, "d")

        self._initial_function = ["0"]
        self._initial_derivative =  ["0"]

        self._time_interval = (0, 1)
        self._space_interval = (0, 1)

        self._dx = 10**-3
        self._dt = 10**-3

        self._reltol = 10**-6

        self._f = None
        self._J = None

        self.t_vals = None
        self.x_vals = None
        self.u_vals = None

        inputs = ({"pde_str":None}, 
                  {"start_boundry":None, "end_boundry":None}, 
                  {"initial_function":None, "initial_derivative":None}, 
                  {"dx":None, "dt":None},
                  {"time_interval":None, "space_interval":None},
                  {"reltol":None})

        for d in inputs:
            for kwarg in d:
                try:
                    d[kwarg] = kwargs[kwarg]
                except KeyError:
                    pass 
        
        funcs = (self.set_pde,
                 self.set_boundry,
                 self.set_initial_condition,
                 self.set_step_size,
                 self.set_interval,
                 self.set_tolerance)

        for idx, func in enumerate(funcs):
            temp_kwargs = inputs[idx]
            func(**temp_kwargs)

    def solve(self):
        """Discretizes in space using the finite difference method and then solves the resulting system through time with the middlepoint method."""
        if self._pde_tokens is None:
            raise TypeError("PDE not found. Use set_pde(pde_str) to input a PDE.")

        start_time = self._time_interval[0]
        end_time = self._time_interval[1]
        start_place = self._space_interval[0]
        end_place = self._space_interval[1]

        self._match_dx_to_n_x()

        n_t = math.ceil((end_time - start_time)/self._dt) + 1
        n_x = round((end_place - start_place)/self._dx) + 1

        v0 = self._get_initial_condition()

        t_vals = np.zeros((1, n_t))
        t_temp = start_time
        for i in range(n_t):
            t_vals[0, i] = t_temp
            t_temp += self._dt

        u_vals = np.zeros((n_x, n_t))

        x_vals = np.linspace(start_place, end_place, n_x)

        for i in range(1, n_x-1):
            u_vals[i, 0] = v0[i, 0]

        v0 = self._resize_v0(v0)

        self._insert_solution(v0, 0, t_vals, u_vals)

        expressions = self._get_derivative_coefficient_expressions()
        coefficients = self._evaluate_derivative_coefficient_expressions(expressions, t_eval=start_time)

        if not self._A_time_dependent:
            self._A = self._get_linear_matrix(coefficients)
            self._A = sp.csr_matrix(self._A)
            is_sparse = True

        if not self._g_time_dependent:
            self._g = self._get_linear_forcing_term(coefficients, t_eval=None)

        for i in range(1, n_t-1):
            if self._linear:
                t_mid = (t_vals[0, i] + t_vals[0, i+1])/2

                if self._A_time_dependent or self._g_time_dependent:
                    new_coefficients = self._evaluate_derivative_coefficient_expressions(expressions, t_eval=t_mid)
                
                if self._A_time_dependent:
                    self._A = self._get_linear_matrix(coefficients)
                    is_sparse = False

                if self._g_time_dependent:
                    self._g = self._get_linear_forcing_term(coefficients, t_eval=t_mid)

                v1 = self._linear_midpoint(t_mid, v0, is_sparse)
            else:
                raise ValueError("Invalid Input: nonlinear PDEs are not yet supported!")

            self._insert_solution(v1, i, t_vals, u_vals)
            v0 = v1

            self.t_vals, self.x_vals, self.u_vals = t_vals, x_vals, u_vals
        return t_vals, x_vals, u_vals

    def max_val(self) -> int|float:
        """Finds the maximum value of u(x,t) in the PDE."""
        if self.u_vals is None:
            raise TypeError("Error: no PDE has been solved yet. Use set_pde(pde_str) and solve() to get u_vals.")
        else:
            max_val = -math.inf
            for i in range(self.u_vals.shape[0]):
                for j in range(self.u_vals.shape[1]):
                    if self.u_vals[i, j] > max_val:
                        max_val = self.u_vals[i, j]
        return float(max_val)

    def min_val(self) -> int|float:
        """Finds the minimum value of u(x,t) in the PDE."""
        if self.u_vals is None:
            raise TypeError("Error: no PDE has been solved yet. Use set_pde(pde_str) and solve() to get u_vals.")
        else:
            min_val = math.inf
            for i in range(self.u_vals.shape[0]):
                for j in range(self.u_vals.shape[1]):
                    if self.u_vals[i, j] < min_val:
                        min_val = self.u_vals[i, j]            
        return float(min_val)

#-----Input methods-----

    def set_pde(self, pde_str:str=None):
        """Sets the PDE to be solved."""
        if pde_str is None:
            return
        self._pde_tokens = self._get_pde_tokens(pde_str)
        self._time_order = self._get_time_order(self._pde_tokens)
        self._linear = self._is_linear(self._pde_tokens)
        if self._linear:
            self._A_time_dependent = self._is_time_dependent("A")
            self._g_time_dependent = self._is_time_dependent("g")
            try:
                for boundry in (self._start_boundry, self._end_boundry):
                    if "t" in boundry[0]:
                        self._g_time_dependent = True
            except TypeError:
                pass
        return 

    def set_boundry(self, start_boundry:tuple|list=None, end_boundry:tuple|list=None):
        """Sets the boundry conditions for the PDE."""
        attributes = ("_start_boundry", "_end_boundry")
        allowed_data_types = (int, float, str,
                              np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64, 
                              np.float16, np.float32, np.float64)

        for i, boundry in enumerate((start_boundry, end_boundry)):
            if type(boundry) is type(None):
                continue
            elif type(boundry) not in (tuple, list):
                raise TypeError("Boundries must be tuples or lists of length 2, e.g. (0, 'd'))")
            elif len(boundry) != 2:
                raise ValueError("Boundry tuples/lists must have length 2.")

            if type(boundry[0]) not in allowed_data_types:
                raise ValueError("First position of boundry tuple must be of type int, float or string.")
            elif type(boundry[0]) is str:
                boundry = (self._get_expression_tokens(boundry[0], "t"), boundry[1])
            elif boundry[1] not in ("d", "n"):
                raise ValueError("Second position of boundry tuple must be strings 'd' or 'n'.")
            
            try:
                if "t" in boundry[0]:
                    self._g_time_dependent = True
            except TypeError:
                pass

            setattr(self, attributes[i], boundry)
        return

    def set_initial_condition(self, initial_function:str|tuple|list=None, initial_derivative:str|tuple|list=None):
        """Sets the initial conditions (u(x,0) and du/dt(x,0)) for the PDE."""
        attributes = ("_initial_function", "_initial_derivative")
        allowed_data_types = (int, float, 
                              np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64, 
                              np.float16, np.float32, np.float64)

        for i, candidate in enumerate((initial_function, initial_derivative)):
            if candidate is None:
                continue
            if type(candidate) is str:
                initial_condition = self._get_expression_tokens(candidate, "x")
            elif type(candidate) in (tuple, list):
                for element in candidate:
                    if type(element) not in allowed_data_types:
                        raise TypeError("Invalid input: elements of input array for initial condition must be int or float.")
                initial_condition = candidate
                warnings.warn("Warning: Make sure the dimensions of your initial condition match the boundry conditions and number of points in space.")
            else:
                raise TypeError("Invalid input: initial conditions must be of type string, list or tuple.")
            setattr(self, attributes[i], initial_condition)
        return

    def set_step_size(self, dx:int|float=None, dt:int|float=None):
        """Sets the step size for either in space (dx), in time (dt) or both."""
        attributes = ("_dx", "_dt")
        errors = ("Step size in space, dx, must be of type int or float.", "Step size in time, dt, must be of type int or float.")
        allowed_data_types = (int, float, 
                              np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64, 
                              np.float16, np.float32, np.float64)        

        for i, step in enumerate((dx, dt)):
            if type(step) is type(None):
                continue
            elif type(step) not in allowed_data_types:
                raise TypeError(errors[i])

            setattr(self, attributes[i], step)
            
            if i == 0:
                self._match_dx_to_n_x()
        return

    def set_interval(self, time_interval:tuple|list=None, space_interval:tuple|list=None):
        """Sets the time interval within which the PDE is to be solved."""
        attributes =("_space_interval", "_time_interval")
        allowed_data_types = (int, float, 
                              np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64, 
                              np.float16, np.float32, np.float64)        
        
        for i, interval in enumerate((space_interval, time_interval)):
            if type(interval) is type(None):
                continue
            elif type(interval) is not tuple:
                raise TypeError("Interval must be a tuple of length 2.")
            elif len(interval) != 2:
                raise ValueError("Interval must have length 2.")
            elif type(interval[0]) not in allowed_data_types or type(interval[1]) not in allowed_data_types:
                raise ValueError("Both positions of the interval tuple must be of type int or float.")
            elif interval[0] > interval[1]:
                raise ValueError("The value at the first position in the tuple must be greater than the value in the second position.")
            
            if i == 0:
                self._match_dx_to_n_x()

            setattr(self, attributes[i], interval)
        return

    def set_tolerance(self, reltol:int|float=None):
        """Sets the relative tolerance for newtons multivariate method when solving non-linear PDE's. The standard relative tolerance is 10^-6."""
        allowed_data_types = (int, float, 
                              np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64, 
                              np.float16, np.float32, np.float64) 
        if reltol is None:
            return

        if type(reltol) not in (int, float):
            raise TypeError("Relative tolerance must be of type int or float.")
        elif reltol < 0:
            raise ValueError("Relative tolerance must be greater than 0.")
        else:
            self._reltol = reltol
        return

#-----Help functions for input methods-----

    def _get_pde_tokens(self, pde_str:str) -> list:
        """Takes a string of correct format and tokenizes constants, variables, differentials and operations. Tokens are stored in an ordered list."""
        num_of_equals = 0
        equals_idx = 0
        for idx, chr in enumerate(pde_str):
            if chr == "=":
                num_of_equals += 1
                equals_idx = idx
            if num_of_equals > 1:
                raise ValueError("Invalid input: only one '=' symbol allowed.")
        if num_of_equals != 1:
            raise ValueError("Invalid input: argument requires at least one '=' symbol.")

        VL = pde_str[0:equals_idx]
        HL = pde_str[equals_idx+1:]

        match_ends = r"[^\+\-\*\/\^=]"
        for side in (VL, HL):
            for i in (0, -1):
                if re.fullmatch(match_ends, side[i]) is None:
                    raise ValueError("Invalid input: '+', '-', '*', '/', '^' and '=' cannot appear at the end of the string or next to '='.")

        match_VL_HL = r"^(?:[\+\-\*\/\^](?![\+\-\*\/\^=])|[0-9]*\.?[0-9]+(?!\.)|(?<![uxtd0-9])[uxt](?![uxtd0-9])|d(2?)ud[xt]\1(?![0-9]))+$"
        if re.fullmatch(match_VL_HL, VL) is None:
            raise ValueError("Invalid input: could not read left side of the equation.")
        elif re.fullmatch(match_VL_HL, HL) is None:
            raise ValueError("Invalid input: could not read right side of the equation.")
        
        patterns = [r"^[\+\-\*\/\^](?![\+\-\*\/\^=])",
                    r"^[0-9]*\.?[0-9]+(?!\.)",
                    r"^(?<![uxtd0-9])[uxt](?![uxtd0-9])",
                    r"^d(2?)ud[xt]\1(?![0-9])"]
        tokens = []

        for side in (VL, HL):
            while len(side) > 0:
                for pattern in patterns:
                    temp_match = re.match(pattern, side)
                    try:
                        side = side.removeprefix(temp_match.group())
                        tokens.append(temp_match.group())                        
                    except (TypeError, AttributeError):
                        continue
            tokens.append("=")
        del tokens[-1] 
        return tokens

    def _get_expression_tokens(self, initial_condition_str:str, var:str) -> list:
        """Takes a string of correct format for either an initial condition expression or a boundry condition expression and tokenizes constants, variables and operations. 
        Tokens are stored in an ordered list."""
        m0 = r"^(?:[\+\-\*\/\^](?![\+\-\*\/\^=])|[0-9]*\.?[0-9]+(?!\.)|(?<![x0-9])[x](?![x0-9]))+$"
        match_function = m0[:61] + var + m0[62:68] + var + m0[69:74] + var + m0[75:]
        match_ends = r"[^\+\-\*\/\^]"

        if re.fullmatch(match_ends, initial_condition_str[0]) is None or re.fullmatch(match_ends, initial_condition_str[-1]) is None:
            raise ValueError("Invalid input: '+', '-', '*', '/' and '^' cannot appear at the end of the expression.")
        elif re.fullmatch(match_function, initial_condition_str) is None:
            raise ValueError("Invalid input: could not interpret initial condition string.")

        patterns = [r"^[\+\-\*\/\^](?![\+\-\*\/\^=])",
                   r"^[0-9]*\.?[0-9]+(?!\.)",
                   r"^" + match_function[56:80]]
        tokens = []

        while len(initial_condition_str) > 0:
            for pattern in patterns:
                temp_match = re.match(pattern, initial_condition_str)
                try:
                    initial_condition_str = initial_condition_str.removeprefix(temp_match.group())
                    tokens.append(temp_match.group())
                except (TypeError, AttributeError):
                    continue
        return tokens

    def _get_time_order(self, pde_tokens:list) -> int:
        """Checks the highest order of the time derivative for the PDE."""
        match_second_order = r"d2udt2"
        match_first_order = r"dudt|d2udxdt"
        first_order = False
        for token in pde_tokens:
            if re.search(match_second_order, token) is not None:
                return 2
            elif re.search(match_first_order, token) is not None:
                first_order = True
        if first_order:
            return 1
        else:
            return 0

    def _separate_into_terms(self, tokens:list) -> list:
        """Takes a PDE token list and returns a list with the PDEs terms as lists nested in a list."""
        parsed_terms = []
        temp_term = []
        #Parse terms in token list
        for token in tokens:
            if token not in ("+", "-", "="):
                temp_term.append(token)
            else:
                parsed_terms.append(temp_term.copy())
                temp_term.clear()
        parsed_terms.append(temp_term)
        return parsed_terms

    def _is_linear(self, tokens:list) -> bool:
        """Checks if the PDE is linear or not."""
        parsed_terms = self._separate_into_terms(tokens)
        match_derivatives = r"d(2?)ud[xt]\1|d2udxdt|u"

        for term in parsed_terms:
            num_of_non_linear_factors = 0
            for idx, token in enumerate(term):
                #Match illegal derivative factors
                if re.fullmatch(match_derivatives, token) is not None:
                    num_of_non_linear_factors += 1
                    #Match illegal exponents of derivative factor
                    try:
                        if term[idx+1] == "^":
                            return False
                    except IndexError:
                        pass
                elif token == "/":
                    if re.fullmatch(match_derivatives, term[idx+1]) is not None:
                        return False
            if num_of_non_linear_factors > 1:
                return False
        return True

    def _is_time_dependent(self, object_type:str) -> bool:
        """Takes a PDE token list and either checks if the resulting matrix or resulting forcing term used for solving in time would be time dependent.
        Returns True if the A or g is time dependent; returns False otherwise."""
        parsed_terms = self._separate_into_terms(self._pde_tokens)

        for term in parsed_terms:           
            if "t" not in term:
                continue
            contains_derivative = False 
            for token in term:
                if token in ("dudx", "dudt", "d2udx2", "d2udt2", "u") and object_type == "A":
                    return True
                elif token in ("dudx", "dudt", "d2udx2", "d2udt2", "u") and object_type == "g":
                    contains_derivative = True
            if object_type == "g" and not contains_derivative:
                return True
        return False

#-----Help functions for solving linear PDEs-----

    def _match_dx_to_n_x(self):
        """Modifies dx so that the number of steps is a whole number, preserving the boundry interval. Returns the number of intervals given the step size as an int."""
        start = self._space_interval[0]
        stop = self._space_interval[1]
        n = round((stop-start)/self._dx)
        if n == 0:
            raise ValueError("Invalid input: for space_interval = (start, stop), (stop - start)/dx must be larger than 1!")
        self._dx = (stop-start)/n
        return

    def _get_initial_condition(self):
        """Takes the tokenized lists for the initial conditions and creates a numpy array with the systems initial values."""
        start = self._space_interval[0]
        stop = self._space_interval[1]
        n_x = round((stop-start)/self._dx) + 1

        x_vals = np.linspace(start, stop, n_x)
        v0 = np.zeros((2*n_x, 1)) if self._time_order == 2 else np.zeros((n_x, 1)) 

        if type(self._initial_function[0]) is str:
            for idx, x_val in enumerate(x_vals):
                v0[idx, 0] = self._evaluate(self._initial_function, x_val)
        else:
            for idx in range(n_x):
                v0[idx, 0] = self._initial_function[idx]
        
        if self._time_order == 2:
            if type(self._initial_derivative[0]) is str:
                for idx, x_val in enumerate(x_vals):
                    v0[idx + n_x, 0] = self._evaluate(self._initial_derivative, x_val)
            else:
                for idx in range(n_x):
                    v0[idx + n_x, 0] = self._initial_derivative[idx]
        return v0

    def _resize_v0(self, v0):
        """Resizes v0 when when we have Dirichlet boundries."""
        if self._time_order == 1:
            if self._start_boundry[1] == "d":
                v0 = v0[1:]
            if self._end_boundry[1] == "d":
                v0 = v0[:-1]
        if self._time_order == 2:
            n = int(len(v0)/2)
            v = v0[:n+1]
            w = v0[n+1:]
            if self._start_boundry[1] == "d":
                v = v[1:]
                w = w[1:]
                n -= 1
            if self._end_boundry[1] == "d":
                v = v[:-1]
                w = w[:-1]
                n -= 1
            new_v0 = np.zeros((2*n, 1))
            for idx, v_comp in enumerate(v):
                new_v0[idx, 0] = v_comp
            for idx, w_comp in enumerate(w):
                new_v0[idx+n, 0] = w_comp
            v0 = new_v0
        return v0

    def _insert_solution(self, v, t_idx:int, t_vals, u_vals):
        """Help function which inserts boundry values in solution matrix at a certain point in time t0."""
        end_range = u_vals.shape[0]
        start_idx = 0
        if self._start_boundry[1] == "d":
            try:
                if type(self._start_boundry[0][0]) is str:
                    u_vals[0, t_idx] = self._evaluate(self._start_boundry[0], t_vals[0, t_idx])
            except TypeError:
                u_vals[0, t_idx] = self._start_boundry[0]
            start_idx += 1
            end_range -= 1
        if self._end_boundry[1] == "d":
            try:
                if type(self._end_boundry[0][0]) is str:
                    u_vals[-1, t_idx] = self._evaluate(self._end_boundry[0], t_vals[0, t_idx])
            except TypeError:
                u_vals[-1, t_idx] = self._end_boundry[0]
            end_range -= 1

        for i in range(0, end_range):
            u_vals[start_idx + i, t_idx] = v[i, 0]
        return
    
    def _evaluate(self, tokens_0:list, var_eval:int|float) -> int|float:
        """Evaluates an expression with correct syntax at a specific value for var_eval."""
        tokens = tokens_0.copy()
        #Evaluate x     
        for idx, token in enumerate(tokens):
            if token == "x" or token == "t":
                tokens[idx] = str(var_eval)
        #Evaluate exponents
        parsed_list = []
        prev_idx = 0
        for idx, token in enumerate(tokens):
            if token == "^":
                base = tokens[idx-1]
                exponent = tokens[idx+1]
                parsed_list += tokens[prev_idx:idx-1] + [str(float(base)**float(exponent))]
                prev_idx = idx + 2
        parsed_list = parsed_list + tokens[prev_idx:]
        tokens = parsed_list
        #Evaluate divisions and multiplications
        parsed_list = []
        prev_idx = 0
        temp_product = 0
        multiplying = False

        def _mult_or_div(factor_1:int|float, operator:str, factor_2:int|float) -> int|float:
            if operator == "*":
                return factor_1*factor_2
            elif operator == "/":
                return factor_1/factor_2

        for idx, token in enumerate(tokens):
            if token in ("*", "/") and not multiplying:
                parsed_list += tokens[prev_idx:idx-1]
                temp_product = _mult_or_div(float(tokens[idx-1]), token, float(tokens[idx+1]))
                multiplying = True
            elif token in ("*", "/") and multiplying:
                factor = float(tokens[idx+1])
                temp_product = _mult_or_div(temp_product, token, factor)
            elif token in ("+", "-") and multiplying:
                prev_idx = idx 
                multiplying = False
                parsed_list += [str(temp_product)]
                temp_product = 0 
        if multiplying:
            parsed_list += [str(temp_product)]
        else:
            parsed_list += tokens[prev_idx:]
        tokens = parsed_list
        #Evaluate addition and subtraction
        if tokens[0] in ("-", "+"):
            tokens_sum = 0
        else:
            tokens_sum = float(tokens[0])
        for idx, token in enumerate(tokens):
            if token == "+":
                tokens_sum += float(tokens[idx+1])
            elif token == "-":
                tokens_sum -= float(tokens[idx+1])
        return tokens_sum

    def _strip_factor(self, term:list, factor:str) -> list:
        """Takes a term and removes factor from the term while preserving the operation syntax in the term token list."""
        for idx, token in enumerate(term):
            if token == factor:
                term[idx] = "1"
        return term

    def _get_derivative_coefficient_expressions(self) -> tuple[list|None, list|None, list|None, list|None, list|None, list|None]:
        """Given a linear PDE of time order max=2 and no mixed terms, you can rewrite it as: c1(x,t)*dudx + c2(x,t)*d2udx2 + c3(x,t)*dudt + c4(x,t)*d2udt2 c5(x,t)*u + r(x,t) = 0. This function returns
        expressions for c1, c2, c3, c4, c5 and r in the form of lists. If cx/r is zero, then return None."""
        parsed_terms = []
        temp_term = []
        #Parse terms in token list
        at_VL = True
        for token in self._pde_tokens:
            if token not in ("+", "-", "="):
                temp_term.append(token)
            else:
                parsed_terms.append(temp_term.copy())
                temp_term.clear()
                if at_VL:
                    if token == "-" or token == "=":
                        temp_term.append("-")
                    if token == "=":
                        at_VL = False
                elif not at_VL:
                    if token == "+":
                        temp_term.append("-")
        parsed_terms.append(temp_term)
        
        coeff_dict = {"dudx":[],   #c1(x,t)
                      "d2udx2":[], #c2(x,t)
                      "dudt":[],   #c3(x,t)
                      "d2udt2":[], #c4(x,t)
                      "u":[]}      #c5(x,t)
        r = []                     #r(x,t)

        for term in parsed_terms:
            key_not_found = True
            for key in coeff_dict:
                if key in term:
                    term = self._strip_factor(term, key)
                    if term[0] not in ("+", "-"):
                        coeff_dict[key] += ["+"] + term
                    else:
                        coeff_dict[key] += term
                    key_not_found = False
                    break
            if key_not_found:
                if term[0] not in ("+", "-"):
                    r += ["+"] + term
                else:
                    r += term
        ans = []
        for key in coeff_dict:
            ans.append(coeff_dict[key])
        ans.append(r)
        return tuple(ans)

    def _evaluate_derivative_coefficient_expressions(self, expressions:tuple, t_eval:int|float|None=None):
        start = self._space_interval[0]
        stop = self._space_interval[1]
        n_x = round((stop-start)/self._dx) + 1
        x_vals = np.linspace(start, stop, n_x)

        coefficients = []
        for expr in expressions:
            temp_expr = expr.copy()
            if temp_expr == []:
                coefficients.append(np.zeros((n_x, 1)))
                continue
            v_eval = np.zeros((n_x, 1))
            for idx, token in enumerate(temp_expr):
                if token == "t":
                    temp_expr[idx] = t_eval
            for idx, x_val in enumerate(x_vals):
                v_eval[idx, 0] = self._evaluate(temp_expr, x_val)
            coefficients.append(v_eval.copy())
        return tuple(coefficients)

    def _get_linear_matrix(self, coefficients:tuple):
        """Creates the coefficient matrix A (from iterative formula for integrating in time: v_k+1 = v_k + h*f(1/2(v_k+1 + v_k) = v_k + h*(A*(1/2(v_k+1 + v_k)) + g(1/2(t_k+1 + t_k))). 
        Is used when the PDE is linear."""
        c1, c2, c3, c4, c5, _ = coefficients

        start = self._space_interval[0]
        stop = self._space_interval[1]
        d = round((stop-start)/self._dx) + 1 #Number of inner points (dimension of A)

        for boundry in (self._start_boundry, self._end_boundry):
            if boundry[1] == "d":
                d -= 1

        A = np.zeros((2*d, 2*d)) if self._time_order == 2 else np.zeros((d, d))

        # Get A

        if self._start_boundry[1] == "d":
            m = 1
        elif self._start_boundry[1] == "n":
            m = 0

        if self._time_order == 1:
            for i in range(1, d-1):
                A[i, i-1] = (c1[i, 0]/(2*self._dx) - c2[i, 0]/(self._dx**2))/c3[i, 0]
                A[i, i] = (2*c2[i, 0]/(self._dx**2) - c5[i, 0])/c3[i, 0]
                A[i, i+1] = (-c1[i, 0]/(2*self._dx) - c2[i, 0]/(self._dx**2))/c3[i, 0]

            if self._start_boundry[1] == "d":
                A[0, 0] = (2*c2[1, 0]/(self._dx)**2 - c5[1, 0])/c3[1, 0]
                A[0, 1] = (-c1[1, 0]/(2*self._dx) - c2[1, 0]/(self._dx)**2)/c3[1, 0]

            elif self._start_boundry[1] == "n":
                A[0, 0] = (2*c2[0, 0]/(self._dx)**2 - c5[0, 0])/c3[0, 0]
                A[0, 1] = (-2*c2[0, 0]/(self._dx)**2)/c3[0, 0]

            if self._end_boundry[1] == "d": 
                A[-1, -2] = (c1[-2, 0]/(2*self._dx) - c2[-2, 0]/(self._dx)**2)/c3[-2, 0]
                A[-1, -1] = (2*c2[-2, 0]/(self._dx)**2 - c5[-2, 0])/c3[-2, 0]
            if self._end_boundry[1] == "n":
                A[d-1, d-2] = (-2*c2[-1, 0]/(self._dx)**2)/c3[-1, 0]
                A[d-1, d-1] = (2*c2[-1, 0]/(self._dx)**2 - c5[-1, 0])/c3[-1, 0]

        elif self._time_order == 2:
            for i in range(1, d-1):
                A[i, i+d] = 1

                A[i+d, i-1] = (c1[i, 0]/(2*self._dx) - c2[i, 0]/(self._dx**2))/c4[i, 0]
                A[i+d, i] = (2*c2[i, 0]/(self._dx**2) - c5[i, 0])/c4[i, 0]
                A[i+d, i+1] = (-c1[i, 0]/(2*self._dx) - c2[i, 0]/(self._dx**2))/c4[i, 0]
                
                A[i+d, i+d] = -c3[i, 0]/c4[i, 0]

            A[0, d] = 1
            A[d-1, d-1 + d] = 1

            if self._start_boundry[1] == "d":
                A[d, 0] = (2*c2[1, 0]/(self._dx)**2 - c5[1, 0])/c4[1, 0]
                A[d, 1] = (-c1[1, 0]/(2*self._dx) - c2[1, 0]/(self._dx)**2)/c4[1, 0]

                A[d, d] = -c3[1, 0]/c4[1, 0]
            elif self._start_boundry[1] == "n":
                A[d, 0] = (2*c2[0, 0]/(self._dx)**2 - c5[0, 0])/c4[0, 0]
                A[d, 1] = (-2*c2[0, 0]/(self._dx)**2)/c4[0, 0]

                A[d, d] = -c3[0, 0]/c4[0, 0]

            if self._end_boundry[1] == "d":
                A[d+d-1, d-2] = (c1[-2, 0]/(2*self._dx) - c2[-2, 0]/(self._dx)**2)/c4[-2, 0]
                A[d+d-1, d-1] = (2*c2[-2, 0]/(self._dx)**2 - c5[-2, 0])/c4[-2, 0]

                A[d+d-1, d+d-1] = -c3[-2, 0]/c4[-2, 0]
            elif self._end_boundry[1] == "n":
                A[d+d-1, d-2] = (-2*c2[-1, 0]/(self._dx)**2)/c4[-1, 0]
                A[d+d-1, d-1] = (2*c2[-1, 0]/(self._dx)**2 - c5[-1, 0])/c4[-1, 0]

                A[d+d-1, d+d-1] = -c3[-1, 0]/c4[-1, 0]
        return A 

    def _get_linear_forcing_term(self, coefficients:tuple, t_eval:int|float|None=None):
        """Returns a forcing term g(t) (from iterative formula for integrating in time: v_k+1 = v_k + h*f(1/2(v_k+1 + v_k) = v_k + h*(A*(1/2(v_k+1 + v_k)) + g(1/2(t_k+1 + t_k))).
        Is used when the PDE is linear."""
        c1, c2, c3, c4, _, r = coefficients

        start = self._space_interval[0]
        stop = self._space_interval[1]
        d = round((stop-start)/self._dx) + 1 #Number of inner points (dimension of g)

        for boundry in (self._start_boundry, self._end_boundry):
            if boundry[1] == "d":
                d -= 1

        g = np.zeros((2*d, 1)) if self._time_order == 2 else np.zeros((d, 1))

        try:
            if type(self._start_boundry[0][0]) == str:
                alfa = self._evaluate(self._start_boundry[0], t_eval)
        except TypeError:
            alfa = self._start_boundry[0]

        try:
            if type(self._end_boundry[0][0]) == str:
                beta = self._evaluate(self._end_boundry[0], t_eval)
        except TypeError:
            beta = self._end_boundry[0]

        if self._time_order == 1:
            for i in range(1, d-1):
                g[i, 0] = r[i, 0]/c3[i, 0]

            if self._start_boundry[1] == "d":
                g[0, 0] = (c1[1, 0]/(2*self._dx) - c2[1, 0]/(self._dx)**2)/c3[1, 0]*alfa + r[1, 0]/c3[1, 0]
            elif self._start_boundry[1] == "n":
                g[0, 0] = (-c1[0, 0]*alfa + 2*c2[0, 0]*alfa/self._dx + r[0, 0])/c3[0, 0]
            
            if self._end_boundry[1] == "d":
                g[-1, 0] = (-c1[-2, 0]/(2*self._dx) - c2[-2, 0]/(self._dx)**2)/c3[-2, 0]*beta + r[-2, 0]/c3[-2, 0]
            elif self._end_boundry[1] == "n":
                g[-1, 0] = (-c1[-1, 0]*beta - 2*c2[-1, 0]*beta/self._dx + r[-1, 0])/c3[-1, 0]

        if self._time_order == 2:
            for i in range(1, d-1):
                g[i+d, 0] = r[i, 0]/c4[i, 0]

            if self._start_boundry[1] == "d":
                g[d, 0] = (c1[1,0]/(2*self._dx) - c2[1, 0]/(self._dx)**2)/c4[1, 0]*alfa + r[1, 0]/c4[1, 0]
            elif self._start_boundry[1] == "n":
                g[d, 0] = (-c1[0, 0]*alfa + 2*c2[0, 0]*alfa/self._dx + r[0, 0])/c4[0, 0]

            if self._end_boundry[1] == "d":
                g[-1, 0] = (-c1[-2, 0]/(2*self._dx) - c2[-2, 0]/(self._dx)**2)/c4[-2, 0]*beta + r[-2, 0]/c4[-2, 0]
            elif self._end_boundry[1] == "n":
                g[-1, 0] = (-c1[-1, 0]*beta - 2*c2[-1, 0]*beta/self._dx + r[-1, 0])
        return g

    def _linear_midpoint(self, t_mid:int|float, v0, is_sparse:bool):
        """Returns the value of v_{k+1} = (u_1(t_{k+1}, u_2...)), where u_1, ... are inner points in the space interval, given a v_k. 
        The implicit midpoint method is used for stability."""
        d = self._A.shape[0]
        h = self._dt
        A = self._A
        g = self._g

        if is_sparse:
            I = sp.identity(d, format="csr")
            v1 = sp.linalg.spsolve(I - h/2*A, (I + h/2*A)@v0 + h*g)
            v1 = v1.reshape((-1, 1))
        else:
            I = np.identity(d)
            v1 = np.linalg.solve(I - h/2*A, (I + h/2*A)@v0 + h*g)

        return v1

#-----Methods for non linear PDEs-----

    def _insert_boundry_condition(self, f):
        """Takes f if non-linear and augments it according to the boundry conditions (either Dirichlet or Neumann)."""
        return f

    def _replace_with_stencil(tokens:list, dx:float, dt:float) -> list:
        """Takes a token list and replaces all of the space differentials with approximate central difference stencils. Now the token list represents
        components of a vector, where each component solves the PDE in time at a certain position."""
        return tokens

    def _get_diff_eq_vector(tokens:list):
        """Takes a tokenlist with defined vector components and creates a whole vector where each component represents the time evolution of a discrete point. 
        Is used when the PDE is non-linear."""
        return f #numpy vector

    def _get_jacobian(F): #F is a numpy vector
        """Takes a vector F and creates its Jacobian matrix."""
        return J #numpy matrix
        
    def _replace_second_time_derivative(tokens:list):
        """Takes a differential equation and creates a system of differential equations in order to replace the second time derivative."""
        return tokens_1, tokens_2 #tokens_1 are for the first m components and tokens_2 are for the components from m+1 to 2m

    def _differentiate(f:str, diff_variable: str) -> str:
        """Takes a function f and differentiates it. If it is not an explicit function, then return the notation for a differential of a higher order."""
        return f_prim

    def _chain_rule(f:str, diff_variable:str) -> str:
        """Implements the chain rule."""
        return f_prim