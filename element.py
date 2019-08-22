from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol
from os import listdir, system
from os.path import join, exists, split
import numpy as np

# path to the data folder
__pth__ = join(split(__file__)[0], 'data')
# retrieve all the symbolic matrices
__sources__ = [f for f in listdir(__pth__) if f.lower().endswith(".py")]
__matrices__ = {f[:-3]:f[:-3]+".k" for f in __sources__}

for py, m in zip(__sources__, __matrices__):
    if not exists(join(__pth__, m)):
        # create .k (symbolic matrix file) if necessary
        system('python "' + join(__pth__, py) + '"')

def available_elements():
    '''
    the list of all available element types
    '''
    return list(__matrices__.keys())
    
def load_symbolic_k(type):
    '''
    load stiffness matrix k of given type of element
    '''
    with open(join(__pth__,__matrices__[type])) as file:
        expr = file.read()
    # a sympy object (Matrix)
    return parse_expr(expr)

def make_numerical_k(matrix, dict):
    '''
    convert the symbolic matrix to numerical one by substitution
    '''
    sym = {str(s):s for s in matrix.free_symbols}
    dict_sub = {sym[k]:dict[k] for k in dict if k in sym.keys()}
    arr = np.array(matrix.subs(dict_sub),dtype=np.float64)
    arr[np.abs(arr)<1e-8]=0    
    return arr
    
def stiffness_matrix(type, Em, nu, a=0.5,b=0.5,c=0.5):
    '''
    retrieve numerical stiffness matrix by specifying element type, elastic modulus and Poisson's ratio
    
    by default, the volume of the element is 1 (a=b=c=0.5)
    '''
    ke_symbolic = load_symbolic_k(type)
    return make_numerical_k(ke_symbolic, {"a":a, "b":b, "c":c, "Em":Em, "nu":nu})