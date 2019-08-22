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
        a, b, c, x, y, z = symbols('a b c x y z')
        E, nu = symbols('Em nu') # use Em not E to prevent confusion with sympy.numbers.exp1
        
        G = E / (2 * (1 + nu))
        g = E /  ((1 + nu) * (1 - 2 * nu))

        o = symbols('o') #  dummy symbol
        
        xlist = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
        ylist = [y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y]
        zlist = [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z]
        yxlist = [y, x, o, y, x, o, y, x, o, y, x, o, y, x, o, y, x, o, y, x, o, y, x, o]
        zylist = [o, z, y, o, z, y, o, z, y, o, z, y, o, z, y, o, z, y, o, z, y, o, z, y]
        zxlist = [z, o, x, z, o, x, z, o, x, z, o, x, z, o, x, z, o, x, z, o, x, z, o, x]

        # Shape functions:
        N1 = (a - x) * (b - y) * (c - z) / (8 * a * b * c)
        N2 = (a + x) * (b - y) * (c - z) / (8 * a * b * c)
        N3 = (a + x) * (b + y) * (c - z) / (8 * a * b * c)
        N4 = (a - x) * (b + y) * (c - z) / (8 * a * b * c)
        N5 = (a - x) * (b - y) * (c + z) / (8 * a * b * c)
        N6 = (a + x) * (b - y) * (c + z) / (8 * a * b * c)
        N7 = (a + x) * (b + y) * (c + z) / (8 * a * b * c)
        N8 = (a - x) * (b + y) * (c + z) / (8 * a * b * c)

        # Create strain-displacement matrix B:
        B0 = map_(diff, [N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0, 0], xlist)
        B1 = map_(diff, [0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8, 0], ylist)
        B2 = map_(diff, [0, 0, N1, 0, 0, N2, 0, 0, N3, 0, 0, N4, 0, 0, N5, 0, 0, N6, 0, 0, N7, 0, 0, N8], zlist)
        B3 = map_(diff, [N1, N1, N1, N2, N2, N2, N3, N3, N3, N4, N4, N4, N5, N5, N5, N6, N6, N6, N7, N7, N7, N8, N8, N8], yxlist)
        B4 = map_(diff, [N1, N1, N1, N2, N2, N2, N3, N3, N3, N4, N4, N4, N5, N5, N5, N6, N6, N6, N7, N7, N7, N8, N8, N8], zylist)
        B5 = map_(diff, [N1, N1, N1, N2, N2, N2, N3, N3, N3, N4, N4, N4, N5, N5, N5, N6, N6, N6, N7, N7, N7, N8, N8, N8], zxlist)
        B = Matrix([B0, B1, B2, B3, B4, B5])

        # Create constitutive (material property) matrix for plane stress:
        C = Matrix([[(1 - nu) * g, nu * g, nu * g, 0, 0, 0],
                    [nu * g, (1 - nu) * g, nu * g, 0, 0, 0],
                    [nu * g, nu * g, (1 - nu) * g, 0, 0, 0],
                    [0, 0, 0,                      G, 0, 0],
                    [0, 0, 0,                      0, G, 0],
                    [0, 0, 0,                      0, 0, G]])
            
        dK = B.T * C * B
        K = dK.integrate((x, -a, a),(y, -b, b),(z, -c, c))
        
        with open(fname,"w") as file:
            file.write(str(K))

symbolic_create()