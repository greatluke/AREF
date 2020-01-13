import constants as const
import math
zp = 1#charge number of cation
zm = -1#charge number of anion
epsilon = 60#permittivity
H = 3e-5#electrode spacing
f = 1e3#applied frequency
Dp = 5e-8#mobility of cation
Dm = 1.5e-7#mobility of anion
V0 = 100#applied potential
np_inf = 3e23#bulk concentration of cation
nm_inf = 3e23#bulk concentration of anion
N = int(1e5) # number of cell
dt = 0.00001
gamma = 0.5
tolerance = 1e-6

class Params():
    def __init__(self):
        self.v0 = V0/const.Vt 
        self.delta = Dm/Dp 
        self.Ld = math.sqrt(math.sqrt(Dm*Dp)/f)/H 
        self.n0 = np_inf*(zm**2*zp-zp**2*zm)
        self.kappaH = (epsilon*const.epsilon_0*const.kB*const.T/(self.n0*const.e**2))**(-0.5)*H
        self.zp = zp 
        self.zm = zm
        self.dt = dt 
        self.dy = 1/N 
        self.num_cell = N
        self.gamma = gamma
        self.tolerance = tolerance 

""" params = Params()
print(params.kappaH)
print(params.Ld) """