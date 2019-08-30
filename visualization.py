# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
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

class Visualize2d_Dynamic():
    '''
    2d visualization of design variable
    '''
    def __init__(self,topo,size=5):
        self.topo = topo
        self.fig = plt.figure(figsize=(size,topo.num_element_y / topo.num_element_x * size))
        self.ax = self.fig.gca()
        
        self.fig.show()
        
    def visualize(self):
        topo=self.topo
        if topo.dof==2:
            self.ax.cla() # clean the axes
            arr = topo.design_var.reshape(topo.num_element_y, topo.num_element_x)
            # flip the first axis so that visually the y direction is up
            self.ax.imshow(np.flip(arr,axis=0),cmap="binary")
            plt.draw()
            plt.pause(0.001) # let the GUI run
            
    def visualize_keep(self):
        '''
        keep showing the figure, the program will not exit until user close the window
        '''
        plt.show()
            
class Visualize3d():
    def __init__(self, topo):
        self.topo = topo
        # plt.ion()
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.gca(projection='3d')
        #self.ax.set_aspect('equal')
        
        self.fig.show()
        # plt.show()
        
    def visualize(self):
        topo=self.topo
        self.ax.cla() # clean the axes
        #self.ax.set_aspect('equal')
        
        voxels = topo.design_var.reshape(topo.num_element_z, topo.num_element_y, topo.num_element_x).T
        voxels_flat = np.sort(voxels.flatten())
        thres = voxels_flat[int(len(voxels_flat) * (1 - topo.target_volume_ratio))]
        
        voxels_bin = voxels>=thres
        
        colors = np.empty(voxels.shape, dtype=object)
        
        for val in range(19):
            val_f = (1+val)*0.05
            colors[voxels>val_f] = str(1-val_f)
            
        self.ax.voxels(voxels_bin, facecolors=colors)
        plt.draw()
        plt.pause(0.001) # let the GUI run
        
    def visualize_keep(self):
        '''
        keep showing the figure, the program will not exit until user close the window
        '''
        plt.show()