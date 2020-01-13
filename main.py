import cupy as cp 
from cupyx.scipy.sparse.linalg import lsqr
import constants as const
import matplotlib.pyplot as plt  
import input, continuity_nm, continuity_np, gauss_eq 
import time, math 
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()
def check_CFL(V,N,z,D,dt,h):
    u = cp.zeros(N-1)
    u[:] = z*D*(V[2:N+1]-V[0:N-1])/(h*const.Vt)
    maxCFL = max(abs(u*dt/h))
    print("MaxCFL = ",maxCFL)

params = input.Params()
num_cell = params.num_cell 

gauss = gauss_eq.Gauss(params)
cont_nm = continuity_nm.Continuity(params)
cont_np = continuity_np.Continuity(params)
""" oldnp = cp.zeros(num_cell+1); newnp = cp.zeros(num_cell+1)
oldnm = cp.zeros(num_cell+1); newnm = cp.zeros(num_cell+1) """
oldnp = input.np_inf/params.n0*cp.ones(num_cell+1); newnp = cp.zeros(num_cell+1)
oldnm = input.nm_inf/params.n0*cp.ones(num_cell+1); newnm = cp.zeros(num_cell+1)
oldv = cp.zeros(num_cell+1); newv = cp.zeros(num_cell+1)
t = 0
dt = params.dt

v_leftBC = math.sin(2*math.pi*t); v_rightBC = 0
oldv[0] = v_leftBC; oldv[-1] = v_rightBC
start = time.time()

while t<1:
    not_converged = False
    t += dt  
    print(t)
    v_leftBC = math.sin(2*math.pi*t)
    #while error_np > params.tolerance:
    gauss.set_rhs(oldnp, oldnm, v_leftBC, v_rightBC, params)
    newv = lsqr((1-params.gamma)*gauss.M,-params.gamma*gauss.M.dot(oldv)+gauss.rhs)[0]
    #check_CFL(newv,num_cell,params.zp,input.Dp,dt,params.dy)
    check_CFL(newv,num_cell,params.zm,input.Dm,dt,params.dy)
    cont_np.setup_matrix(params, newnp)
    newnp = lsqr(cont_np.M_exp,cont_np.M_imp.dot(oldnp))[0]
    
    cont_nm.setup_matrix(params, newnm)
    newnm = lsqr(cont_nm.M_exp,cont_nm.M_imp.dot(oldnm))[0]

    oldv=newv; oldnp=newnp; oldnm=newnm
    print(oldv)
end = time.time()
print(f"Total time: {endtime-start}") 



        

