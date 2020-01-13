from numba import cuda 
import cupy as cp 
import math
from cupyx.scipy.sparse import diags

class Continuity():
    def __init__(self, params):
        num_cell = params.num_cell
        self.c1 = params.Ld**2/math.sqrt(params.delta)/params.dy**2
        self.c2 = params.zp*params.v0
        self.lower_diag = cp.zeros(num_cell)
        self.main_diag = cp.zeros(num_cell+1)
        self.upper_diag = cp.zeros(num_cell) 
        """ self.M_imp = diags([self.lower_diag,self.main_diag,self.upper_diag],[-1,0,1])
        self.M_exp = diags([self.lower_diag,self.main_diag,self.upper_diag],[-1,0,1]) """

    def setup_matrix(self, params, v):
        self.main_diag[1:-1] = self.c1*(-2+self.c2/2*(v[2:]-2*v[1:-1]+v[0:-2]))
        self.lower_diag[:-1] = self.c1*(1-self.c2/2*(v[1:-1]-v[0:-2]))
        self.upper_diag[1:] = self.c1*(1+self.c2/2*(v[2:]-v[1:-1]))
        self.main_diag[0] = self.c1*(-2+self.c2*(v[1]-v[0]))
        self.main_diag[-1] = -self.c1*(2+self.c2*(v[-1]-v[-2]))
        self.lower_diag[-1] = -self.c1*(-2+self.c2*(v[-1]-v[-2]))
        self.upper_diag[0] = self.c1*(2+self.c2*(v[1]-v[0]))
        self.M_imp = diags([params.gamma*self.lower_diag,1/params.dt+params.gamma*self.main_diag,params.gamma*self.upper_diag],[-1,0,1])
        self.M_exp = diags([-(1-params.gamma)*self.lower_diag,1/params.dt-(1-params.gamma)*self.main_diag,-(1-params.gamma)*self.upper_diag],[-1,0,1])
