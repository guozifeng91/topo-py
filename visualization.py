import matplotlib.pyplot as plt
import numpy as np
from os.path import join

class Visualize2d_C():
    '''
    2d visualization of elemental strain energy
    '''
    def __init__(self,topo,save_to = None, filename=None, size = 5, dpi = 300):
        self.topo = topo
        self.save_to = save_to
        self.filename = filename
        self.count = 0
        self.size = size
        self.dpi = dpi
        
    def visualize(self):
        topo=self.topo
        if topo.dof==2:
            arr = topo.c_elements.reshape(topo.num_element_y, topo.num_element_x)
            
            plt.figure(figsize=(self.size,topo.num_element_y / topo.num_element_x * self.size))
            # flip the first axis so that visually the y direction is up
            plt.imshow(np.flip(arr,axis=0))
            
            if self.save_to is None or self.filename is None:
                plt.show()
            else:
                plt.savefig(join(self.save_to,self.filename + str(self.count) + ".png"),dpi=self.dpi, pad_inches=0, bbox_inches='tight')
                self.count += 1
            plt.close()
            
class Visualize2d():
    '''
    2d visualization of design variable
    '''
    def __init__(self,topo,save_to = None, filename=None, size = 5, dpi = 300):
        self.topo = topo
        self.save_to = save_to
        self.filename = filename
        self.count = 0
        self.size = size
        self.dpi = dpi
        
    def visualize(self):
        topo=self.topo
        if topo.dof==2:
            arr = topo.design_var.reshape(topo.num_element_y, topo.num_element_x)
            
            plt.figure(figsize=(self.size,topo.num_element_y / topo.num_element_x * self.size))
            # flip the first axis so that visually the y direction is up
            plt.imshow(np.flip(arr,axis=0),cmap="binary")
            
            if self.save_to is None or self.filename is None:
                plt.show()
            else:
                plt.savefig(join(self.save_to,self.filename + str(self.count) + ".png"),dpi=self.dpi, pad_inches=0, bbox_inches='tight')
                self.count += 1
            plt.close()
            
class Visualize3d():
    def __init__(self, topo):
        pass