import numpy as np
import element as ele

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# solid and void status (design variable) of elements
SOLID=1.0

VOID=1e-8
    
class Topology():
    '''
    topology with homologous square/cubic elements
    
    usage:
    
    1. creating the object
    top = Topology(ke)
    top.set_domain()
    
    2. setting boundary conditions (in any sequence)
    top.set_load_dof_range()
    top.set_fix_dof_range()
    top.set_actv_pasv_elements_range()
    
    3. running optimization
    top.SIMP()
    
    4. get the results
    '''
    def __init__(self, ke):
        '''
        input: the elemental stiffness matrix ke, this class assumes all elements are the same (size, material)
        '''
        self.ke = ke
        # initialize some variables
        self.c_elements = None # sensitivity number of elements
        self.c_elements_last = None # sensitivity number of elements in the last iteration (for BESO method only)
        
        # actv and pasv elements (for SIMP method only)
        self.pasv_elements = np.array([],dtype=np.int32)
        self.actv_elements = np.array([],dtype=np.int32)
    
    def set_domain(self, num_element_x, num_element_y, num_element_z=0):
        self.num_element_x = num_element_x
        self.num_element_y = num_element_y
        self.num_element_z = num_element_z
        self.dof = 2 if num_element_z == 0 else 3 # degree of freedom, currently just 2 or 3
        
        self.element_num = num_element_x * num_element_y * (1 if num_element_z < 1 else num_element_z)
        self.node_num = (self.num_element_x + 1) * (self.num_element_y + 1) * (self.num_element_z + 1)
        self.matrix_size = self.dof * self.node_num # the size of the global stiffness matrix

        # the element-node topology (which element contains which nodes)
        
        # Nodes are numbered as follows.
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
        if self.dof == 2:
            self.element2node = [[self.get_node_index(x,y),self.get_node_index(x+1,y),self.get_node_index(x+1,y+1),self.get_node_index(x,y+1)] for y in range(num_element_y) for x in range(num_element_x)]
        else:
            # x in the inner loop, z in the outer loop
            self.element2node = [[ self.get_node_index(x,y,z),self.get_node_index(x+1,y,z),self.get_node_index(x+1,y+1,z),self.get_node_index(x,y+1,z),
                                    self.get_node_index(x,y,z+1),self.get_node_index(x+1,y,z+1),self.get_node_index(x+1,y+1,z+1),self.get_node_index(x,y+1,z+1)] for z in range(num_element_z) for y in range(num_element_y) for x in range(num_element_x)]
        self.element2node = np.array(self.element2node)
        
        # the element-dof topology (which element contains which dof)
        # e.g, dof=2, element2node[0]=[0,1,7,6], meaning the first element contains nodes [0,1,7,6]
        # after the following calculation, the dof_indice[0] will be [0,1,2,3,14,15,12,13]
        # meaning the the first element contains dof (x and y translations) [0,1,2,3,14,15,12,13]
        
        node_indice = self.element2node
        to_shape = list(self.element2node.shape)
        to_shape[1] *= self.dof
        
        if self.dof == 2:
            self.element2node_dof = np.transpose(np.array([node_indice*self.dof, node_indice*self.dof+1]),[1,2,0]).reshape(to_shape)
        else:
            self.element2node_dof = np.transpose(np.array([node_indice*self.dof, node_indice*self.dof+1,node_indice*self.dof+2]),[1,2,0]).reshape(to_shape)
    
    def set_fix_dof_range(self, x, y, z):
        '''
        set fixed degree of freedom
        
        x, y, z are lists of the bounding box, e.g. x = [[min_u,max_u,min_v,max_v,min_w,max_w], [min_u,max_u,min_v,max_v,min_w,max_w],...]
        
        where min and max are node coordinate within the mesh domain, max is excluded
        '''
        x_list=[]
        y_list=[]
        z_list=[]
        if self.dof==2:
            for min_u,max_u,min_v,max_v in x:
                x_list = x_list + [self.get_node_index(u,v) for v in range(min_v,max_v) for u in range(min_u, max_u)]
                
            for min_u,max_u,min_v,max_v in y:
                y_list = y_list + [self.get_node_index(u,v) for v in range(min_v,max_v) for u in range(min_u, max_u)]
        else:
            for min_u,max_u,min_v,max_v,min_w,max_w in x:
                x_list = x_list + [self.get_node_index(u,v,w) for w in range(min_w,max_w) for v in range(min_v,max_v) for u in range(min_u, max_u)]
                
            for min_u,max_u,min_v,max_v,min_w,max_w in y:
                y_list = y_list + [self.get_node_index(u,v,w) for w in range(min_w,max_w) for v in range(min_v,max_v) for u in range(min_u, max_u)]
                
            for min_u,max_u,min_v,max_v,min_w,max_w in z:
                z_list = z_list + [self.get_node_index(u,v,w) for w in range(min_w,max_w) for v in range(min_v,max_v) for u in range(min_u, max_u)]

        x_list = list(set(x_list))
        y_list = list(set(y_list))
        z_list = list(set(z_list))
        
        self.set_fix_dof(x_list,y_list,z_list)
        
    def set_fix_dof(self, x, y, z):
        '''
        set fixed degree of freedom
        
        x, y, z are lists of the indexes of nodes that are fixed on respective directions
        '''
        if self.dof==2:
            self.fix_dof = np.sort(np.concatenate([np.array(x,dtype=np.int32)*self.dof,np.array(y,dtype=np.int32)*self.dof+1]))
        else:
            self.fix_dof = np.sort(np.concatenate([np.array(x,dtype=np.int32)*self.dof,np.array(y,dtype=np.int32)*self.dof+1,np.array(z,dtype=np.int32)*self.dof+2]))
        
        self.free_dof = np.setdiff1d(np.arange(self.matrix_size,dtype=np.int32), self.fix_dof)
        
        # the location of each global free dof in row-reduced matrix
        # e.g. free_dof=[1,3,5,6,8], then 1->0, 3->1, 5->2, ...
        # this array provide a fast way to do row reduction of global K
        self.dof_row_in_reduced_k = np.arange(self.matrix_size)
        self.dof_row_in_reduced_k[self.free_dof] = np.arange(len(self.free_dof))
        self.dof_row_in_reduced_k[self.fix_dof] = self.matrix_size # fix_dof set to out-of-bound
    
    def set_load_dof_range(self, x_node, y_node, z_node, x_load, y_load, z_load):
        '''
        set load degree of freedom, by giving ranges of x, y, z node
        
        e.g. x_node = [[min_u,max_u,min_v,max_v,min_w,max_w], [min_u,max_u,min_v,max_v,min_w,max_w],...]
        
        length of x_node and x_load must be equal, same for y and z
        
        currently no duplicate element is allowed
        '''
        x_node_list=[]
        y_node_list=[]
        z_node_list=[]
        
        x_load_list=[]
        y_load_list=[]
        z_load_list=[]
        
        if self.dof==2:
            for node,load in zip(x_node,x_load):
                min_u,max_u,min_v,max_v = node
                x_node_list = x_node_list + [self.get_node_index(u,v) for v in range(min_v,max_v) for u in range(min_u, max_u)]
                x_load_list = x_load_list + ([load]* (max_u - min_u) * (max_v - min_v))
                
            for node,load in zip(y_node,y_load):
                min_u,max_u,min_v,max_v = node
                y_node_list = y_node_list + [self.get_node_index(u,v) for v in range(min_v,max_v) for u in range(min_u, max_u)]
                y_load_list = y_load_list + ([load]* (max_u - min_u) * (max_v - min_v))
        else:
            for node,load in zip(x_node,x_load):
                min_u,max_u,min_v,max_v,min_w,max_w = node
                x_node_list = x_node_list + [self.get_node_index(u,v,w) for w in range(min_w,max_w) for v in range(min_v,max_v) for u in range(min_u, max_u)]
                x_load_list = x_load_list + ([load] * (max_u - min_u) * (max_v - min_v) * (max_w - min_w))
                
            for node,load in zip(y_node,y_load):
                min_u,max_u,min_v,max_v,min_w,max_w = node
                y_node_list = y_node_list + [self.get_node_index(u,v,w) for w in range(min_w,max_w) for v in range(min_v,max_v) for u in range(min_u, max_u)]
                y_load_list = y_load_list + ([load] * (max_u - min_u) * (max_v - min_v) * (max_w - min_w))
                
            for node,load in zip(z_node,z_load):
                min_u,max_u,min_v,max_v,min_w,max_w = node
                z_node_list = z_node_list + [self.get_node_index(u,v, w) for w in range(min_w,max_w) for v in range(min_v,max_v) for u in range(min_u, max_u)]
                z_load_list = z_load_list + ([load] * (max_u - min_u) * (max_v - min_v) * (max_w - min_w))

        self.set_load_dof(x_node_list,y_node_list,z_node_list,x_load_list,y_load_list,z_load_list)
        
    def set_load_dof(self, x_node, y_node, z_node, x_load, y_load, z_load):
        '''
        set load degree of freedom, by giving indices (x_node, y_node, z_node) of x, y and z dof
        '''
        # load vector
        self.load_vector = np.zeros(self.matrix_size, dtype=np.float64)
        self.load_vector[np.array(x_node,dtype=np.int32)*self.dof] = np.array(x_load,dtype=np.float64)
        self.load_vector[np.array(y_node,dtype=np.int32)*self.dof+1] = np.array(y_load,dtype=np.float64)
        if self.dof==3:
            self.load_vector[np.array(z_node,dtype=np.int32)*self.dof] = np.array(z_load,dtype=np.float64)
    
    def set_actv_pasv_elements_range(self, actv_elements, pasv_elements):
        '''
        set active and passive element by giving [[min_u,max_u,min_v,max_v,min_w,max_w],[min_u,max_u,min_v,max_v,min_w,max_w],...]
        '''
        psv_elem=[]
        if self.dof==2:
            for min_u,max_u,min_v,max_v in pasv_elements:
                psv_elem = psv_elem + [self.get_element_index(u,v) for v in range(min_v,max_v) for u in range(min_u, max_u)]
        else:
            for min_u,max_u,min_v,max_v,min_w,max_w in pasv_elements:
                psv_elem = psv_elem + [self.get_element_index(u,v,w) for w in range(min_w,max_w) for v in range(min_v,max_v) for u in range(min_u, max_u)]
        psv_elem = list(set(psv_elem))
        
        atv_elem=[]
        if self.dof==2:
            for min_u,max_u,min_v,max_v in actv_elements:
                atv_elem = atv_elem + [self.get_element_index(u,v) for v in range(min_v,max_v) for u in range(min_u, max_u)]
        else:
            for min_u,max_u,min_v,max_v,min_w,max_w in actv_elements:
                atv_elem = atv_elem + [self.get_element_index(u,v,w) for w in range(min_w,max_w) for v in range(min_v,max_v) for u in range(min_u, max_u)]
        atv_elem = list(set(atv_elem))
        
        self.set_actv_pasv_elements(atv_elem, psv_elem)
        
    def set_actv_pasv_elements(self, actv_elements, pasv_elements):
        '''
        set active and passive element by giving element indices
        '''
        if pasv_elements is not None:
            self.pasv_elements = np.array(pasv_elements,dtype=np.int32)
        else:
            self.pasv_elements = np.array([],dtype=np.int32)
            
        if actv_elements is not None:
            self.actv_elements = np.array(actv_elements,dtype=np.int32)
        else:
            self.actv_elements = np.array([],dtype=np.int32)
        
        itsc=np.intersect1d(self.pasv_elements, self.actv_elements)
        if len(itsc)>0:
            print("warning, passive and active elements overlaps", itsc)

    def get_node_index(self, x, y, z=0):
        '''
        get the index of the node at x, y, z location
        '''
        return (self.num_element_x + 1) * (self.num_element_y + 1) * z + (self.num_element_x + 1) * y + x
    
    def get_element_index(self, x, y, z=0):
        '''
        get the index of the element at x, y, z location
        '''
        return self.num_element_x * self.num_element_y * z + self.num_element_x * y + x

    def reset_design_var_BESO(self):
        self.design_var = np.ones(self.element_num, dtype=np.float64)
        
    def reset_design_var_SIMP(self):
        self.design_var = 0.5*np.ones(self.element_num, dtype=np.float64)
        
        self.design_var[self.pasv_elements]=VOID
        self.design_var[self.actv_elements]=SOLID
        
    def update_K_sparse_BESO(self):
        '''
        update global stiffness matrix
        '''
        # use three array to store the row, col and data that will use to build sparse matrix
        
        ke_length=self.ke.shape[0]**2 # number of elements in ke, used at the step counter
        length=ke_length * self.element_num # the length of each array is determined
        
        rows=np.zeros(length,dtype=np.int32)
        cols=np.zeros(length,dtype=np.int32)
        data=np.zeros(length,dtype=np.float64)

        pos=0 # pointer
        
        # for each element
        for i in range(self.element_num):
            ke = (self.ke * self.design_var[i]).flatten() # update the local k based on the design variable (which has been initialized)
            element_dof = self.element2node_dof[i] # the position of each row/col of ke in global K
            ones = np.ones_like(element_dof)
            
            global_row_ke=np.einsum("i,j->ij",element_dof,ones).flatten() # the rows of each element of ke in global K (2d array flattened to 1d)
            global_col_ke=np.einsum("i,j->ji",element_dof,ones).flatten() # the columns of each element of ke in global K (2d array flattened to 1d)
            
            # copy the array to the corresponding position of the large array
            rows[pos:pos+ke_length]=global_row_ke
            cols[pos:pos+ke_length]=global_col_ke
            data[pos:pos+ke_length]=ke
            
            pos += ke_length
        
        # create sparse matrix
        # similar with coo matrix (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix)
        # duplicate indices are summed, which is exactly what we need
        self.K_sparse = csr_matrix((data, (rows, cols)), shape=(self.matrix_size, self.matrix_size))
        
        # construct row-reduced Ke here as using self.K_sparse[self.free_dof][:,self.free_dof] to do row reduction is very slow
        # data that both row and col are in free_dof
        mask = np.logical_and(np.isin(rows,self.free_dof), np.isin(cols,self.free_dof)) 

        rows_reduc=rows[mask]
        cols_reduc=cols[mask]
        data_reduc=data[mask]
        
        # change global indice to reduced indice (e.g. free_dof=[1,3,5], then row 1 of global K will be row 0 of reduced K, same as 3->1,5->2)
        rows_reduc=self.dof_row_in_reduced_k[rows_reduc]
        cols_reduc=self.dof_row_in_reduced_k[cols_reduc]

        self.K_sparse_reduction = csr_matrix((data_reduc, (rows_reduc, cols_reduc)), shape=(len(self.free_dof), len(self.free_dof)))

    def update_K_sparse_SIMP(self, p=3):
        '''
        update global stiffness matrix
        
        p is the penalty factor, which is commonly chosen as 3 
        '''
        # use three array to store the row, col and data that will use to build sparse matrix
        
        ke_length=self.ke.shape[0]**2 # number of elements in ke, used at the step counter
        length=ke_length * self.element_num # the length of each array is determined
        
        rows=np.zeros(length,dtype=np.int32)
        cols=np.zeros(length,dtype=np.int32)
        data=np.zeros(length,dtype=np.float64)

        pos=0 # pointer
        
        # for each element
        for i in range(self.element_num):
            # for SIMP method, the Elastic modulus of less dense material is E0 = E * rhp^p
            # when a rho_min is chose, the Ke is (rho_min + (1-rho_min)rho^p)ke
            # (see http://help.solidworks.com/2019/english/SolidWorks/cworks/c_simp_method_topology.htm)
            ke = (self.ke * (VOID + (SOLID-VOID)*self.design_var[i]**p)).flatten() # update the local k based on the design variable (interpreted as density by SIMP)
            
            element_dof = self.element2node_dof[i] # the position of each row/col of ke in global K
            ones = np.ones_like(element_dof)
            
            global_row_ke=np.einsum("i,j->ij",element_dof,ones).flatten() # the rows of each element of ke in global K (2d array flattened to 1d)
            global_col_ke=np.einsum("i,j->ji",element_dof,ones).flatten() # the columns of each element of ke in global K (2d array flattened to 1d)
            
            # copy the array to the corresponding position of the large array
            rows[pos:pos+ke_length]=global_row_ke
            cols[pos:pos+ke_length]=global_col_ke
            data[pos:pos+ke_length]=ke
            
            pos += ke_length
        
        # create sparse matrix
        # similar with coo matrix (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix)
        # duplicate indices are summed, which is exactly what we need
        self.K_sparse = csr_matrix((data, (rows, cols)), shape=(self.matrix_size, self.matrix_size))
        
        # construct row-reduced Ke here as using self.K_sparse[self.free_dof][:,self.free_dof] to do row reduction is very slow
        # data that both row and col are in free_dof
        mask = np.logical_and(np.isin(rows,self.free_dof), np.isin(cols,self.free_dof)) 

        rows_reduc=rows[mask]
        cols_reduc=cols[mask]
        data_reduc=data[mask]
        
        # change global indice to reduced indice (e.g. free_dof=[1,3,5], then row 1 of global K will be row 0 of reduced K, same as 3->1,5->2)
        rows_reduc=self.dof_row_in_reduced_k[rows_reduc]
        cols_reduc=self.dof_row_in_reduced_k[cols_reduc]

        self.K_sparse_reduction = csr_matrix((data_reduc, (rows_reduc, cols_reduc)), shape=(len(self.free_dof), len(self.free_dof)))
        
    def fea_sparse(self):
        '''
        sparse matrix implementation of fea()
        
        calculation seems within reasonable time
        '''
        # self.update_K_sparse()

        # conventional row reduction, very slow (no matter what format K_sparse is)
        # K_reduction = self.K_sparse[self.free_dof][:,self.free_dof]
        # K_reduction = K_reduction.tocsr()
        
        # use sparse row-reduced K
        K_reduction = self.K_sparse_reduction
        load_reduction = self.load_vector[self.free_dof]

        # solve the linear system (this take the most time of fea, permc_spec='NATURAL' seems faster, 50% than default setting)
        disp_reduction = spsolve(K_reduction, load_reduction, use_umfpack=True, permc_spec='NATURAL') # Force = K * Disp

        # retrieve the result
        self.disp_vector = np.zeros(self.matrix_size,dtype=np.float64)
        self.disp_vector[self.free_dof] = disp_reduction
        
        self.force_vector = self.K_sparse.tocsr().dot(self.disp_vector)
        self.reaction_vector = self.force_vector - self.load_vector

    def update_K_BESO(self):
        '''
        dense implementation of generating global stiffness matrix
        '''
        # global stiffness matrix, will be replaced by sparse matrix implementation
        self.K = np.zeros(shape = (self.matrix_size, self.matrix_size), dtype=np.float64)
        
        for i in range(self.element_num):
            ke = (self.ke * self.design_var[i]).flatten() # update the local k based on the design variable (which has been initialized)
            element_dof = self.element2node_dof[i] # the position of each row/col of ke in global K
            ones = np.ones_like(element_dof)
            
            global_row_ke=np.einsum("i,j->ij",element_dof,ones).flatten() # the rows of each element of ke in global K (2d array flattened to 1d)
            global_col_ke=np.einsum("i,j->ji",element_dof,ones).flatten() # the columns of each element of ke in global K (2d array flattened to 1d)
        
            self.K[global_row_ke,global_col_ke] += ke
            
        # fill the stiffness matrix (deprecated code, easy to understand but slow)
        #for i in range(self.element_num):
        #    ke = self.ke * self.design_var[i] # update the local k based on the design variable (which has been initialized)
        #    nodes = self.element2node[i] # for each elements (the nodes of each element)
        #    
        #    # assemble the local k to the global K
        #    for node_a in range(len(nodes)):
        #        coord_local_a = node_a * self.dof
        #        coord_global_a = nodes[node_a] * self.dof
        #        for node_b in range(len(nodes)):
        #            coord_local_b = node_b * self.dof
        #            coord_global_b = nodes[node_b] * self.dof
        #            self.K[coord_global_a:coord_global_a+self.dof,coord_global_b:coord_global_b+self.dof] += ke[coord_local_a:coord_local_a+self.dof,coord_local_b:coord_local_b+self.dof]
    
    def update_K_SIMP(self,p=3):
        '''
        dense implementation of generating global stiffness matrix
        '''
        # global stiffness matrix, will be replaced by sparse matrix implementation
        self.K = np.zeros(shape = (self.matrix_size, self.matrix_size), dtype=np.float64)
        
        for i in range(self.element_num):
            # for SIMP method, the Elastic modulus of less dense material is E0 = E * rhp^p
            # when a rho_min is chose, the Ke is (rho_min + (1-rho_min)rho^p)ke
            # (see http://help.solidworks.com/2019/english/SolidWorks/cworks/c_simp_method_topology.htm)
            ke = (self.ke * (VOID+(SOLID-VOID)*self.design_var[i]**p)).flatten() # update the local k based on the design variable
            
            element_dof = self.element2node_dof[i] # the position of each row/col of ke in global K
            ones = np.ones_like(element_dof)
            
            global_row_ke=np.einsum("i,j->ij",element_dof,ones).flatten() # the rows of each element of ke in global K (2d array flattened to 1d)
            global_col_ke=np.einsum("i,j->ji",element_dof,ones).flatten() # the columns of each element of ke in global K (2d array flattened to 1d)
        
            self.K[global_row_ke,global_col_ke] += ke
    
    def fea(self):
        '''
        finite element analysis implemented by dense matrix
        
        the code is slow and memory consuming, and should be used only for validating the results
        '''
        # remove rows and columns of fixed dof (this code is not optimized so it is slow)
        K_reduction = self.K[self.free_dof][:,self.free_dof]

        load_reduction = self.load_vector[self.free_dof]
        
        # solve the linear system
        disp_reduction = np.linalg.solve(K_reduction, load_reduction) # Force = K * Disp
        
        # retrieve the result
        self.disp_vector = np.zeros(self.matrix_size,dtype=np.float64)
        self.disp_vector[self.free_dof] = disp_reduction
        
        self.force_vector = np.matmul(self.K, self.disp_vector)
        self.reaction_vector = self.force_vector - self.load_vector

    def sensitivity_analysis_BESO(self):
        '''
        sensitivity calculation:
        
        c = 1/2 u_t.ke.u (eq. 3.3 of Huang & Xie, EVOLUTIONARY TOPOLOGY OPTIMIZATION OF CONTINUUM STRUCTURES)
        '''
        self.c_elements_last = self.c_elements # store the last sensitivity numbers
        
        # displacement of each element, shape=(num_elem, dof_per_elem)
        u = self.disp_vector[self.element2node_dof]

        # do matmul for each element of a with b
        # result shape = (num_elem, dof_per_elem)
        ut_ke = np.einsum("ijk,kj->ij",u[:,None],self.ke) # equivalent to: np.tensordot(u, self.ke, [1,0]), or [np.matmul(u_,ke) for u_ in u]

        # do dot product for each element of ut_ke with each element of a
        # result shape = (num_elem,)
        # note that the ke is different for each element (linear correlation with design variable)
        ut_ke_u = 0.5 * np.einsum("ij,ij->i",ut_ke,u) * self.design_var # equivalent to: ut_ke_u = 1/2 * np.sum(ut_ke * u,axis=1) * self.design_var

        self.c_elements = ut_ke_u # sensitivity (elemental strain energy)
    
    def sensitivity_analysis_SIMP(self, p=3):
        '''
        calculate the objective (sum of the element strain energy) and the sensitivity (derivative of objective)
        
        design_var is interpreted as the density of the material
        
        p is the penalty factor, according to http://help.solidworks.com/2019/english/SolidWorks/cworks/c_simp_method_topology.htm, p=3 is common
        '''
        # displacement of each element, shape=(num_elem, dof_per_elem)
        u = self.disp_vector[self.element2node_dof]

        # do matmul for each element of a with b
        # result shape = (num_elem, dof_per_elem)
        ut_ke = np.einsum("ijk,kj->ij",u[:,None],self.ke) # equivalent to: np.tensordot(u, self.ke, [1,0]), or [np.matmul(u_,ke) for u_ in u]

        # do dot product for each element of ut_ke with each element of a
        # result shape = (num_elem,)
        # note that the ke is different for each element (linear correlation with design variable)
        ut_ke_u = np.einsum("ij,ij->i",ut_ke,u) # equivalent to: ut_ke_u = np.sum(ut_ke * u,axis=1)
        
        # the sensitivity is the derivative of the objective function
        self.c_elements = -p * ut_ke_u * (self.design_var ** (p- 1))
        
        self.C = (ut_ke_u * (self.design_var ** p)).sum()
    
    def filter_sensitivity_SIMP(self, r_min):
        '''
        filtering method by Sigmund 1997, Sigmund&Maute 2012
        
        c is the sensitivity number, w is the weights, rho is the density, i is the neighbors, e is the current element, r_min is the predefined radius
        
        c_e_new = sum(c_i * w_i * rho_i) / (rho_e * sum(w_i))
        
        where w_i = r_min - dist_i_e
        '''
        c_elements_new = np.zeros(self.element_num,dtype=np.float64)
        r_min_int = int(np.floor(r_min))
        
        if self.dof==2: # for each element
            U,V=np.indices([self.num_element_x,self.num_element_y])
            for i in range(self.num_element_x):
                # min max indices of nodes
                u_min = np.maximum(0, i - r_min_int) # min indice, included
                u_max = np.minimum(self.num_element_x, i + r_min_int + 1) # max indice, excluded
                for j in range(self.num_element_y):
                    v_min = np.maximum(0, j - r_min_int)
                    v_max = np.minimum(self.num_element_y, j + r_min_int + 1)

                    self_index =self.get_element_index(i,j)
                    
                    u = U[u_min:u_max,v_min:v_max].flatten() # 1d array of u indices of elements
                    v = V[u_min:u_max,v_min:v_max].flatten() # 1d array of v indices of elements
                    
                    element_indices = v*(self.num_element_x)+u
                    
                    # 2d distance
                    w_element = r_min - np.sqrt((u - i)**2 + (v - j)**2) # 1d array of w_element = r_min - dist
                    w_element[w_element<0] = 0
                    
                    #sum(c_i * w_i * rho_i)
                    sum_top=(w_element * self.c_elements[element_indices] * self.design_var[element_indices]).sum()
                    
                    #(rho_e * sum(w_i))
                    sum_bottom = w_element.sum() * self.design_var[self_index]
                    
                    c_elements_new[self_index] =  sum_top / sum_bottom
        else:
            U,V,W=np.indices([self.num_element_x,self.num_element_y,self.num_element_z])
            for i in range(self.num_element_x):
                # min max indices of nodes
                u_min = np.maximum(0, i - r_min_int) # min indice, included
                u_max = np.minimum(self.num_element_x, i + r_min_int + 1) # max indice, excluded
                for j in range(self.num_element_y):
                    v_min = np.maximum(0, j - r_min_int)
                    v_max = np.minimum(self.num_element_y, j + r_min_int + 1)
                    for k in range(self.num_element_z):
                        w_min = np.maximum(0, k - r_min_int)
                        w_max = np.minimum(self.num_element_z, k + r_min_int + 1)
                        
                        self_index =self.get_element_index(i,j,k)
                        
                        u = U[u_min:u_max,v_min:v_max,w_min:w_max].flatten() # 1d array of u indices of elements
                        v = V[u_min:u_max,v_min:v_max,w_min:w_max].flatten() # 1d array of v indices of elements
                        w = W[u_min:u_max,v_min:v_max,w_min:w_max].flatten() # 1d array of v indices of elements
                        
                        element_indices = w*self.num_element_x*self.num_element_y + v*self.num_element_x + u
                    
                        # 3d distance
                        w_element = r_min - np.sqrt((u-i)**2 + (v-j)**2 + (w-k)**2) # 1d array of w_element = r_min - dist
                        w_element[w_element<0] = 0
                    
                        #sum(c_i * w_i * rho_i)
                        sum_top=(w_element * self.c_elements[element_indices] * self.design_var[element_indices]).sum()
                        
                        #(rho_e * sum(w_i))
                        sum_bottom = w_element.sum() * self.design_var[self_index]
                        
                        c_elements_new[self_index] =  sum_top / sum_bottom
                        
        self.c_elements = c_elements_new
        
    def filter_sensitivity_BESO(self, r_min):
        '''
        filtering (kind of smoothing) of sensitivity numbers
        
        stages: 1. calculate the nodal sensitivity from the elements; 2. update elemental sensitivity by nodal; 3. average the new with the old
        '''
        # step one, get node sensitivity by averaging element sensitivity
        node2element_count=np.zeros(self.node_num,dtype=np.uint8)
        c_nodes=np.zeros(self.node_num,dtype=np.float64)
        c_elements_new = np.zeros(self.element_num,dtype=np.float64)
        
        for nodes,c in zip(self.element2node,self.c_elements):
            node2element_count[nodes] +=1
            c_nodes[nodes] += c # the elements are assumed squared / cubic, therefore all weights are 1

        # sensitivity number of nodes
        c_nodes /= node2element_count # c_j
        
        r_min_int = int(np.floor(r_min))
        if self.dof==2: # for each element
            U,V=np.indices([self.num_element_x+1,self.num_element_y+1])
            for i in range(self.num_element_x):
                # min max indices of nodes
                u_min = np.maximum(0, i - r_min_int) # min indice, included
                u_max = np.minimum(self.num_element_x, i + r_min_int + 1) # max indice, included (num nodes = num element + 1)
                for j in range(self.num_element_y):
                    v_min = np.maximum(0, j - r_min_int)
                    v_max = np.minimum(self.num_element_y, j + r_min_int + 1)

                    u = U[u_min:u_max+1,v_min:v_max+1].flatten() # 1d array of u indices of nodes
                    v = V[u_min:u_max+1,v_min:v_max+1].flatten() # 1d array of v indices of nodes
                    node_indices = v*(self.num_element_x+1)+u
                    
                    # 2d distance
                    # the (i,j) node is at the corner of the (i,j) element, not the center, therefore -(0.5,0.5)
                    # also, the (i+1,j+1) node is at the corner of (i,j) element as well
                    w_node = r_min - np.sqrt((u - i - 0.5)**2 + (v - j - 0.5)**2) # 1d array of w_node = r_min - r_node2elem
                    w_node[w_node<0] = 0

                    # ci = sum(w_node * c_node) / sum(w_node)
                    # (eq. 3.6 of Huang & Xie, EVOLUTIONARY TOPOLOGY OPTIMIZATION OF CONTINUUM STRUCTURES)
                    c_elements_new[self.get_element_index(i,j)] = (w_node * c_nodes[node_indices]).sum() / w_node.sum()

        else:
            U,V,W=np.indices([self.num_element_x+1,self.num_element_y+1,self.num_element_z+1])
            for i in range(self.num_element_x):
                # min max indices of nodes
                u_min = np.maximum(0, i - r_min_int) # min indice, included
                u_max = np.minimum(self.num_element_x, i + r_min_int + 1) # max indice, included (num nodes = num element + 1)
                for j in range(self.num_element_y):
                    v_min = np.maximum(0, j - r_min_int)
                    v_max = np.minimum(self.num_element_y, j + r_min_int + 1)
                    for k in range(self.num_element_z):
                        w_min = np.maximum(0, k - r_min_int)
                        w_max = np.minimum(self.num_element_z, k + r_min_int + 1)
                        
                        u = U[u_min:u_max+1,v_min:v_max+1,w_min:w_max+1].flatten() # 1d array of u indices of nodes
                        v = V[u_min:u_max+1,v_min:v_max+1,w_min:w_max+1].flatten() # 1d array of v indices of nodes
                        w = W[u_min:u_max+1,v_min:v_max+1,w_min:w_max+1].flatten() # 1d array of w indices of nodes
                        node_indices = w*(self.num_element_x+1)*(self.num_element_y+1) + v*(self.num_element_x+1)+u
                        
                        # 3d distance
                        w_node = r_min - np.sqrt((u - i - 0.5)**2 + (v - j - 0.5)**2 + (w - k - 0.5)**2) # 1d array of w_node = r_min - r_node2elem
                        w_node[w_node<0] = 0

                        # ci = sum(w_node * c_node) / sum(w_node)
                        c_elements_new[self.get_element_index(i,j,k)] = (w_node * c_nodes[node_indices]).sum() / w_node.sum()
        
        # average with last c_elements, as the values for pasv element in c_elements_new are already reset
        if self.c_elements_last is not None:
            # average the current and last sensitivity number
            # (eq. 3.8 of Huang & Xie, EVOLUTIONARY TOPOLOGY OPTIMIZATION OF CONTINUUM STRUCTURES)
            # ci = (ci + ci_last) / 2
            self.c_elements = (c_elements_new + self.c_elements_last) / 2
        else:
            # if there is no last c (1st iteration, average with current c)
            self.c_elements = (c_elements_new + self.c_elements) / 2

    def update_design_var_SIMP(self, volfrac, move=0.1, lam1=0, lam2=1e6, eta = 0.5):
        '''
        OC-like method for updating design variables
        
        code reference:
        https://github.com/vonlippmann/Topology-optimization-of-structure-via-simp-method/blob/Topology_optimization_simp/Python/optimization_simp.py
        https://github.com/williamhunter/topy/blob/master/topy/topology.py
        '''
        
        while (lam2 - lam1) / (lam2 + lam1) > 1e-8 and lam2 > 1e-40:
            lammid = 0.5 * (lam1 + lam2)
            
            B = self.design_var * (-self.c_elements / lammid) ** eta
            design_var_new = np.maximum(VOID, np.maximum(self.design_var - move, np.minimum(SOLID, np.minimum(self.design_var + move, B))))
            
            # passive elements are VOID
            design_var_new[self.pasv_elements]=VOID
            
            # active elements are SOLID
            design_var_new[self.actv_elements] = SOLID
            
            # update lambda
            if self.dof == 2:
                if design_var_new.sum() - self.num_element_x * self.num_element_y * volfrac > 0:
                    lam1 = lammid
                else:
                    lam2 = lammid
            else:
                if design_var_new.sum() - self.num_element_x * self.num_element_y * self.num_element_z * volfrac > 0:
                    lam1 = lammid
                else:
                    lam2 = lammid
        
        self.change = (np.abs(self.design_var - design_var_new)).max()    
        self.design_var = design_var_new
            
    def update_design_var_BESO(self, volume_ratio, max_addition_raio = 0.01):
        target_num_elem = int(self.element_num * (1 - volume_ratio))
        thres = np.sort(self.c_elements)[target_num_elem]
        
        # the indices of void and solid elements
        voids_indices = self.design_var==VOID
        solids_indices = self.design_var==SOLID
        
        thres_add = thres
        thres_del = thres
        # delete elements from solids, and add elements from voids
        add_indice = self.c_elements[voids_indices] >= thres_add
        
        ar = (add_indice * 1).sum()/self.element_num
        if ar > max_addition_raio:
            # adding too many elements at once, recalculate the threshold
            # (section below eq. 3.12 of Huang & Xie, EVOLUTIONARY TOPOLOGY OPTIMIZATION OF CONTINUUM STRUCTURES)
            
            thres_add = np.sort(self.c_elements[voids_indices])[-int(ar * len(voids_indices))]
            thres_del = np.sort(self.c_elements[solids_indices])[int(ar * len(voids_indices))]
            add_indice = self.c_elements[voids_indices] >= thres_add
            
        temp = self.design_var[voids_indices]
        temp[add_indice] = SOLID
        self.design_var[voids_indices] = temp

        delete_indice = self.c_elements[solids_indices] <= thres_del
        temp = self.design_var[solids_indices]
        temp[delete_indice] = VOID
        self.design_var[solids_indices] = temp
        
    def SIMP(self, iteration, volfrac, change_stop=0.01, filter_radius=3, sparse=True, visualizer=[], move=0.1, lam1=0, lam2=1e6, eta=0.5):
        '''
        this implementation follows the SIMP method (which is a gradient based method), taken the following as the reference:
        
        https://github.com/vonlippmann/Topology-optimization-of-structure-via-simp-method
        https://github.com/williamhunter/topy
        http://help.solidworks.com/2019/english/SolidWorks/cworks/c_simp_method_topology.htm
        https://mavt.ethz.ch/content/dam/ethz/special-interest/mavt/department-dam/news/documents/sigmund-presentation-dls-hs-15.pdf
        
        parameters
        -------------
        iteration: number of iterations for optimization
        volfrac: target volume ratio (0 to 1)
        change_stop: optimization stops if the change between 2 iterations less than this value
        filter_radius: filter radius
        sparse: use sparse matrix implementation, note that sparse implementation is optimized and therefore faster than dense version, the dense version should only be use to validate the result
        visualizer: list of visualizer
        
        move: the "learning rate" of the gradient-based process
        '''
        self.reset_design_var_SIMP()
        
        print ("FEA matrix size",len(self.free_dof))
        print ("{:<5} {:<10} {:<10}".format('iter','v_ratio','change'))
        for i in range(iteration):
            if sparse:
                self.update_K_sparse_SIMP()
                self.fea_sparse()
            else:
                self.update_K_SIMP()
                self.fea()

            info = []
            info.append(i)
            
            self.sensitivity_analysis_SIMP()
            
            if filter_radius > 1:
                self.filter_sensitivity_SIMP(filter_radius)

            info.append("%.6f" % (self.design_var.sum()/self.element_num))
            self.update_design_var_SIMP(volfrac, move=move,lam1=lam1,lam2=lam2,eta=eta)
            info.append("%.6f" % self.change)
            print ("{:<5} {:<10} {:<10}".format(info[0],info[1],info[2]))
            
            for vis in visualizer:
                vis.visualize()
                
            if self.change<=change_stop:
                print("converged!")
                break
    
    def BESO(self, iteration, target_volume_ratio, evolution_volume_ratio, converge_thres = 1e-4, max_addition_raio = 0.01, filter_radius=1.5, sparse=True, visualizer=[]):
        '''
        this implementation follows the BESO of EVOLUTIONARY TOPOLOGY OPTIMIZATION OF CONTINUUM STRUCTURES by Huang and Xie
        
        currently the implementation is not fully functional (works for the cantilever and beam cases, when there are just few nodes that have load)
        
        i think i didn't understand the methodology correctly somewhere
        
        parameters
        -------------
        iteration: number of iterations for optimization
        target_volume_ratio: target volume ratio (0 to 1)
        evolution_volume_ratio: change rate of current target volume ratio for each iteration
        converge_thres: after the target_volume_ratio has reached, if the error between recent 10 iterations are less than this value, the optimization stops
        max_addition_raio: maximum ratio of adding elements per iteration
        filter_radius: filter radius
        sparse: use sparse matrix implementation, note that sparse implementation is optimized and therefore faster than dense version, the dense version should only be use to validate the result
        visualizer: list of visualizer
        '''
        self.reset_design_var_BESO()
        N = 5
        C = []
        print ("FEA matrix size",len(self.free_dof))
        print ("{:<5} {:<10} {:<10}".format('iter','v_ratio','err'))
        for i in range(iteration):
            # experimental time cost (matrix size 10200): 0.6851685047149658 0.000997304916381836 0.13663530349731445 0.0009970664978027344
            # stages: fea, sensitivity_analysis, filter_sensitivity, update_design_var
            # therefore fea is the slowest partition (in which solving the linear system is the slowest part)
            
            if sparse:
                self.update_K_sparse_BESO()
                self.fea_sparse()
            else:
                self.update_K_BESO()
                self.fea()

            C.append(np.dot(self.force_vector, self.disp_vector))
            info = []
            info.append(i)
            
            self.sensitivity_analysis_BESO()
            
            if filter_radius > 1:
                self.filter_sensitivity_BESO(filter_radius)

            current_volume_ratio = (self.design_var[self.design_var==SOLID]*1).sum() / len(self.design_var)
            
            if np.abs((current_volume_ratio - target_volume_ratio)/current_volume_ratio) < evolution_volume_ratio:
                next_volume_ratio = target_volume_ratio
            else:
                if current_volume_ratio > target_volume_ratio:
                    next_volume_ratio = current_volume_ratio * (1 - evolution_volume_ratio)
                else:
                    next_volume_ratio = current_volume_ratio * (1 + evolution_volume_ratio)
            
            info.append("%.6f" % next_volume_ratio)
            if next_volume_ratio==target_volume_ratio and i > N * 2:
                sum1 = np.sum(C[-N:])
                sum2 = np.sum(C[-N*2:-N])
                err = np.abs(sum2-sum1)/sum1
                if err <= converge_thres:
                    print("converged!")
                    return
                else:
                    info.append("%.6f" % err)
            else:
                info.append("n/a")
            print ("{:<5} {:<10} {:<10}".format(info[0],info[1],info[2]))
            # print (next_volume_ratio)
            self.update_design_var_BESO(next_volume_ratio, max_addition_raio)

            for vis in visualizer:
                vis.visualize()
    
    def get_result(self):
        return self.design_var