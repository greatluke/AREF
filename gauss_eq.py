from numba import cuda 
import cupy as cp 
from cupyx.scipy.sparse import diags

class Gauss():
    def __init__(self, params):
        num_cell = params.num_cell
        main_diag = -2*cp.ones(num_cell+1)
        lower_diag = cp.ones(num_cell)
        upper_diag = cp.ones(num_cell)
        main_diag[0] = 1
        upper_diag[0] = 0
        main_diag[-1] = 1
        lower_diag[-1] = 0
        self.M = diags([lower_diag,main_diag,upper_diag],[-1,0,1])
        self.rhs = cp.zeros(num_cell+1)
        self.c = -(params.kappaH)**2/params.v0*params.dy**2
    def set_rhs(self, np, nm, v_left_BC, v_right_BC, params):
        self.rhs = (params.zp*np + params.zm*nm)*self.c
        self.rhs[0] = v_left_BC
        self.rhs[-1] = v_right_BC
