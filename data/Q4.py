from sympy import symbols, Matrix, diff, integrate, zeros
import os

# ============================
# === Finite Element Types ===
# ============================

# Nodes are numbered as follows, although this is not important to the user.
#
# 2D: Y             3D: Y
#     |                 |
#   4-|-3             4-|-3
#   | +-|---X        /| +-|---X
#   1---2           / 1/--2
#                  8--/7 /
#                  | / |/
#                  5/--6
#                  /
#                 Z
#

def map_(fun, l1, l2):
    return [fun(v1, v2) for v1,v2 in zip(l1,l2)]
    
def symbolic_create():
    fname = __file__.split(".py")[0] + ".k"
    if not os.path.exists(fname):
        print ("creating symbolic matrix",fname)

        # SymPy symbols:
        a, b, x, y = symbols('a b x y')
        E, nu = symbols('Em nu') # use Em not E to prevent confusion with sympy.numbers.exp1
        # N1, N2, N3, N4 = symbols('N1 N2 N3 N4') # this is actually a useless code
        
        xlist = [x, x, x, x, x, x, x, x]
        ylist = [y, y, y, y, y, y, y, y]
        yxlist = [y, x, y, x, y, x, y, x]

        # Shape functions:
        N1 = (a - x) * (b - y) / (4 * a * b)
        N2 = (a + x) * (b - y) / (4 * a * b)
        N3 = (a + x) * (b + y) / (4 * a * b)
        N4 = (a - x) * (b + y) / (4 * a * b)

        # Create strain-displacement matrix B:
        B0 = map_(diff, [N1, 0, N2, 0, N3, 0, N4, 0], xlist)
        B1 = map_(diff, [0, N1, 0, N2, 0, N3, 0, N4], ylist)
        B2 = map_(diff, [N1, N1, N2, N2, N3, N3, N4, N4], yxlist)
        B = Matrix([B0, B1, B2])

        # Create constitutive (material property) matrix for plane stress:
        C = (E / (1 - nu**2)) * Matrix([[1, nu, 0],
                                        [nu, 1, 0],
                                        [0,  0, (1 - nu) / 2]])
        dK = B.T * C * B
        # for 2d problem, here the material thickness t is assumed to be 1 (the same as the element's dimension)
        K = dK.integrate((x, -a, a),(y, -b, b))
        
        with open(fname,"w") as file:
            file.write(str(K))

symbolic_create()