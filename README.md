# topo-py
a simple SIMP topology optimizer by a non-engineer as a learning project

## Code References
the code cannot run without these projects as references<br>
[topy](https://github.com/williamhunter/topy)<br>
[topology-optimizer](https://github.com/vonlippmann/Topology-optimization-of-structure-via-simp-method)<br>
[topology optimization](https://mavt.ethz.ch/content/dam/ethz/special-interest/mavt/department-dam/news/documents/sigmund-presentation-dls-hs-15.pdf)<br>
[soild work on SIMP](http://help.solidworks.com/2019/english/SolidWorks/cworks/c_simp_method_topology.htm)<br>
[finite element analysis](https://www.taylorfrancis.com/books/9780429453076)

## Some Results
cantilever
![alt text](https://github.com/guozifeng91/topo-py/blob/master/test_cantilever/x39.png)<br><br>
beam
![alt text](https://github.com/guozifeng91/topo-py/blob/master/test_beam/x39.png)<br><br>
tree
![alt text](https://github.com/guozifeng91/topo-py/blob/master/test_tree/x39.png)<br><br>


## Required Libraries
numpy 1.14.2 (for nd-array)<br>
scipy 1.1.0 (for sparse algebra)<br>
sympy 1.4 (for symbolic computation, which computes the stiffness matrix for different types of elements)<br>
matplotlib 3.03 (for visualization)<br>

## Other Things
that comes later
