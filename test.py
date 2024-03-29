from topology import Topology
from visualization import Visualize2d_C, Visualize2d, Visualize3d, Visualize2d_Dynamic
import element
import numpy as np
from multiprocessing import Process
import random
import time

def test_3d(ke):
    top = Topology(ke)
    top.set_domain(20,10,10)
    
    top.set_load_dof_range([],[],[[20,21,0,11,5,6]],[],[],[-1/10])
    top.set_fix_dof_range([[0,1,0,11,0,11]],[[0,1,0,11,0,11]],[[0,1,0,11,0,11]])
    
    visualizer=[Visualize3d(top)]
    top.SIMP(40, 0.3, move=0.1, visualizer=visualizer)
    visualizer[0].visualize_keep()
'''
three test case
'''
def test_tree(ke):
    top = Topology(ke)
    top.set_domain(128,128)
    
    top.set_load_dof_range([],[[0,128,127,128]],None,[],[-1/1000],None)
    top.set_fix_dof_range([[44,49,0,1],[110,115,0,1]],[[44,49,0,1],[110,115,0,1]],None)
    
    # top.set_actv_pasv_elements_range([[0,128,127,128]],[[20,60,40,80]])
    top.set_actv_pasv_elements_range([[0,128,127,128]],[])
    
    visualizer=[Visualize2d_Dynamic(top)]
    top.SIMP(40, 0.3, move=0.2, visualizer=visualizer)
    
    visualizer[0].visualize_keep()
    
def test_beam(ke):
    top = Topology(ke)
    top.set_domain(240,40)
    top.set_fix_dof([0],[0,240],None)
    top.set_load_dof([],[120],None,[],[-1],None)
    # top.BESO(40, 0.5, 0.05, filter_radius=3,sparse=sparse)
    
    visualizer=[Visualize2d(top, "C:\\Users\\guo zifeng\\Desktop\\fea\\topo-py\\test", "x"), Visualize2d_C(top, "C:\\Users\\guo zifeng\\Desktop\\fea\\topo-py\\test", "c")]
    top.SIMP(80, 0.3, move=0.1, visualizer=visualizer)
    
    Visualize2d(top).visualize()
    
def test_cantilever(ke):
    top = Topology(ke)
    top.set_domain(100,50)
    top.set_fix_dof_range([[0,1,0,51]],[[0,1,0,51]],None)
    top.set_load_dof_range([],[[100,101,25,26]],None,[],[-1],None)
    # top.BESO(40, 0.5, 0.05, filter_radius=3, max_addition_raio=0.01,sparse=sparse)
    
    visualizer=[Visualize2d(top, "C:\\Users\\guo zifeng\\Desktop\\fea\\topo-py\\test", "x"), Visualize2d_C(top, "C:\\Users\\guo zifeng\\Desktop\\fea\\topo-py\\test", "c")]
    top.SIMP(40, 0.3, visualizer=visualizer)
    
    Visualize2d(top).visualize()
    
if __name__ == '__main__':
    print ("available elements",element.available_elements())
    
    # optimizing one case is slow, but we can do several in parallel
    
    processes = []
    ke = element.stiffness_matrix("H8", 1, 1/3)
    print(ke)
    ke = element.stiffness_matrix("Q4", 1, 1/3)
    # ke = element.stiffness_matrix("H8", 1, 1/3)
    for m in range(1):
        n = m + 1
        p = Process(target=test_tree, args=[ke])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()