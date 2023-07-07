import numpy as np
from numba import jit
from time import time
from qiskit.quantum_info import DensityMatrix,Statevector,random_statevector
from cirq import is_hermitian,validate_density_matrix,unitary

from dmcx import testfinderhos,tracenumba
from gatematrixs import IdentityGate,cxgate,HadamardGate1Qbit,HadamardGate,XGate

def CompareRHOS(rhoA,rhoB,lindexs):
    print("\n CompareRHOS dif = ",sum(np.abs(rhoA-rhoB)))
    
    for i in range(len(lindexs)):
        x = lindexs[i]        
        print(x,"dif",round(abs(rhoA[x]-rhoB[x]),5))




@jit(nopython = True)   
def evolverho(rho:np.ndarray,gate : np.ndarray):
    gate_transpose = gate.transpose()
    
    # dot same as matmul but for numba
    m = np.dot(gate, rho) 
    m = np.dot(m, np.conjugate(gate_transpose)) 
    return m

def getspacegrid(gridsize,cellsize,centerval):

    # grid size must be odd (impar)
            
    grid = np.zeros(gridsize,dtype=np.float32)
    grid[0]=centerval
    n=1
    for k in range(1,gridsize,2):            
        grid[k] = centerval + n * cellsize
        grid[k+1] = centerval - n * cellsize
        n+=1
    return grid

class gatefinder:
    # gridsize must be odd
    def __init__(self,minval=-1,maxval=1,dims=8,gridsize=5):
                
        
        self.dims = dims        
        self.minval=float(minval)
        self.maxval=float(maxval)
        self.gridsize=gridsize
        self.resetminval=self.minval
        self.resetmaxval=self.maxval                
        self.zoom=0
        self.gate2x2=None
        self.makegrid()
    def makegrid(self):
        self.grid=np.zeros([self.dims,self.gridsize],dtype=np.float32)
        spacesize=self.maxval - self.minval
        self.cellsize=spacesize/(self.gridsize-1)                
        for i in range(self.dims):            
            self.grid[i] =getspacegrid(self.gridsize,self.cellsize,0)
    
    def reset(self):
        self.zoom=0
        self.minval=self.resetminval
        self.maxval=self.resetmaxval
        self.makegrid()
            
            
    def search(self,rhobefore,vecsearch,distbreak=-1):
        self.reset()
        self.gate2x2=None        

        self.mindist=1000000.
        self.searchrecursive(rhobefore,vecsearch,distbreak)
        
        print("\nmindist search",self.mindist)
        return self.gate2x2

    @staticmethod
    @jit(nopython = True)   
    def searchnumba1(rhobefore:np.ndarray, vectorfind: np.ndarray, n:int,grid: np.ndarray , distbreak : int):
        
        I=np.zeros((2,2),dtype=np.complex128)    
        I [0][0]=  1+0j
        I [1][1]=  1+0j 
        gate2x2= np.zeros((2,2),dtype = np.complex128)        
        counter=0
        best=[0,0,0,0,0,0,0,0]        
        mindist=100000.
        for a in range(n):        
            for b in range(n):
                for c in range(n):        
                    for d in range(n):
                        for e in range(n):
                            for f in range(n):
                                for g in range(n):
                                    for h in range(n):
                                        gate2x2[0][0]=grid[0][a] + grid[1][b]*1j
                                        gate2x2[0][1]=grid[2][c] + grid[3][d]*1j
                                        gate2x2[1][0]=grid[4][e] + grid[5][f]*1j
                                        gate2x2[1][1]=grid[6][g] + grid[7][h]*1j
                                        gate2x2 = gate2x2 + np.conjugate(gate2x2.transpose())                                        
                                        gate4x4=np.kron(I,gate2x2) # for qubit 0
                                        rho4x4 = evolverho(rhobefore,gate4x4)
                                        counter+=1                                
                                        dist=0.
                                        for x in range(4):
                                            for w in range(4):
                                                dist += abs(rho4x4[x][w]-vectorfind[x][w])
                                        if dist<mindist:
                                            
                                            best=[a,b,c,d,e,f,g,h]
                                            mindist=dist
                                            if distbreak > -1:
                                                if mindist < distbreak:
                                                    return gate2x2,mindist,best


        gate2x2[0][0]=grid[0][best[0]] + grid[1][best[1]]*1j
        gate2x2[0][1]=grid[2][best[2]] + grid[3][best[3]]*1j
        gate2x2[1][0]=grid[4][best[4]] + grid[5][best[5]]*1j
        gate2x2[1][1]=grid[6][best[6]] + grid[7][best[7]]*1j
        gate2x2 = gate2x2 + np.conjugate(gate2x2.transpose())                        

        return gate2x2,mindist,best

    def searchrecursive(self,rhobefore,vecsearch,distbreak):
        # distbreak= -1 for search all space
        for self.zoom in range(1,100,1):
            
            gate2x2 ,mindist,best = self.searchnumba1(rhobefore,vecsearch, self.gridsize, self.grid,distbreak)

            print("  ",self.zoom,")",round(mindist,5),end="")
            if self.zoom%5==0:  print("")
            if mindist < self.mindist:
                self.gate2x2=gate2x2
                self.mindist=mindist
            
            
            if self.mindist > distbreak:
                oldcellsize=self.cellsize
                self.cellsize *=0.9
                SPACERIGHTLIMIT = 1.
                SPACELEFTLIMIT= -1.
                for i in range(self.dims):
                    idxgrid = best[i]
                    gridcenterval= self.grid[i][idxgrid] + oldcellsize/2
                    if gridcenterval>SPACERIGHTLIMIT-self.cellsize/2:
                        gridcenterval = SPACERIGHTLIMIT-self.cellsize/2
                    if gridcenterval<SPACELEFTLIMIT-self.cellsize/2:
                        gridcenterval = SPACELEFTLIMIT-self.cellsize/2
                        
                    self.grid[i]=getspacegrid(self.gridsize,self.cellsize,gridcenterval)
            else:                
                return
        print("end search by zoom limit 100")

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def testgate2x2(gate2x2,rhobefore,rhoafter):
    print("gate2x2 hermitian = ",is_hermitian(gate2x2),"\ttrace=",np.trace(gate2x2),"\tis_pos_def",is_pos_def(gate2x2))
    gate4x4=np.kron(IdentityGate(),gate2x2) # for qubit 0
    resrho=evolverho(np.copy(rhobefore),gate4x4)
    CompareRHOS(resrho.flatten(),rhoafter.flatten(),np.arange(16))

##gate2x2 = [[ 0.69724107-0.00744126j, -0.34607115-0.01072917j],
##           [ 0.03870212+0.722417j,   -0.17429961+0.82169682j]]
##gate2x2= np.array(gate2x2,dtype=np.complex128)
##gate2x2 = gate2x2 + np.conjugate(gate2x2.transpose())
##print(gate2x2)



idxrho=1
seed=5
rhobefore,rhoafter = testfinderhos(seed=seed, idxrho=idxrho) # cx (0,1) get rho 1 = [0,2]
print("seed =",seed,"idxrho = ",idxrho)
print("rhobefore hermitian = ",is_hermitian(rhobefore),"\ttrace=",np.trace(rhobefore),"\tis_pos_def",is_pos_def(rhobefore))
print("rhoafter hermitian = ",is_hermitian(rhoafter),"\ttrace=",np.trace(rhoafter),"\tis_pos_def",is_pos_def(rhoafter))
##try:    
##    validate_density_matrix(rhoafter,qid_shape=(2,2,))
##except ValueError as e:
##    #  is not positive semidefinite or not hermitian or trace not 1
##    print('Failed validate_density_matrix: ' + str(e))

#testgate2x2(gate2x2,rhobefore,rhoafter)


##sv = random_statevector(4,1)
##rhobefore=DensityMatrix(sv).data
##rhoafter= evolverho(np.copy(rhobefore),cxgate(control=0))
gf= gatefinder()        
resgate2x2= gf.search(rhobefore,rhoafter,0.00001)
print("seed",seed,"idxrho",idxrho)
testgate2x2(resgate2x2,rhobefore,rhoafter)


