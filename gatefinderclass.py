import numpy as np
from numba import jit
from time import time
from qiskit.quantum_info import DensityMatrix,Statevector,random_statevector
from cirq import is_hermitian,validate_density_matrix,unitary

from dmcx import testfinderhos,tracenumba
from gatematrixs import IdentityGate,cxgate,HadamardGate1Qbit,HadamardGate,XGate
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

@jit(nopython = True)
def closest_unitary(A:np.ndarray):
    #https://michaelgoerz.net/notes/finding-the-closest-unitary-for-a-given-matrix/    """

    #V, __, Wh = scipy.linalg.svd(A)
    V, __, Wh = np.linalg.svd(A, full_matrices=False)
    #U = np.matrix(V.dot(Wh))
    U = V.dot(Wh)
    return U



def CompareRHOS(rhoA,rhoB,lindexs):
    print("\n CompareRHOS dif = ",sum(np.abs(rhoA-rhoB)))
    
    for i in range(len(lindexs)):
        x = lindexs[i]        
        print(x,"dif",round(abs(rhoA[x]-rhoB[x]),5))

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



@jit(nopython = True)   
def evolverho(rho:np.ndarray,gate : np.ndarray):
    gate_transpose = gate.transpose()
    
    # dot same as matmul but for numba
    m = np.dot(gate, rho) 
    m = np.dot(m, np.conjugate(gate_transpose)) 
    return m


class gatefinder:
    # gridsize must be odd
    def __init__(self,minval=-1,maxval=1,dims=6,gridsize=7):
                
        
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
        
        I=IdentityGate()
        gate2x2= np.zeros((2,2),dtype = np.complex128)        
        counter=0
        best=[0,0,0,0,0,0]        
        mindist=100000.
        for a in range(n):        
            for b in range(n):
                for c in range(n):        
                    for d in range(n):
                        for e in range(n):
                            for f in range(n):
                                gate2x2[0][0]=grid[0][a] 
                                gate2x2[0][1]=grid[1][b] + grid[2][c]*1j
                                gate2x2[1][0]=grid[3][d] + grid[4][e]*1j
                                gate2x2[1][1]=grid[5][f] 
                                
                                gate2x2=closest_unitary(gate2x2)                                 
                                 # make 4x4 for qbit 0
                                gate4x4=np.kron(I,gate2x2)
                                 # evolve rho
                                rho4x4 = evolverho(rhobefore,gate4x4)
                                counter+=1                                                                        

                                dist = np.abs(rho4x4 - rhoafter).sum()
                                
                                if dist<mindist:
                                    
                                    best=[a,b,c,d,e,f]
                                    mindist=dist
                                    if distbreak > -1:
                                        if mindist < distbreak:
                                            return gate2x2,mindist,best


        gate2x2[0][0]=grid[0][best[0]] 
        gate2x2[0][1]=grid[1][best[1]] + grid[2][best[2]]*1j
        gate2x2[1][0]=grid[3][best[3]] + grid[4][best[4]]*1j
        gate2x2[1][1]=grid[5][best[5]] 
        gate2x2=closest_unitary(gate2x2)                                 

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



def testgate2x2(gate2x2,rhobefore,rhoafter):
    print("gate2x2 hermitian = ",is_hermitian(gate2x2),"\ttrace=",np.trace(gate2x2),"\tis_pos_def",is_pos_def(gate2x2))
    gate4x4=np.kron(IdentityGate(),gate2x2) # for qubit 0
    resrho=evolverho(np.copy(rhobefore),gate4x4)
    CompareRHOS(resrho.flatten(),rhoafter.flatten(),np.arange(16))


                

idxrho=1
#seed=5
seed=7
rhobefore,rhoafter = testfinderhos(seed=seed, idxrho=idxrho) # cx (0,1) get rho 1 = [0,2


##sv = random_statevector(4,1)
##rhobefore=DensityMatrix(sv).data
##rhoafter= evolverho(np.copy(rhobefore),cxgate(control=0))

print("seed =",seed,"idxrho = ",idxrho)
print("rhobefore hermitian = ",is_hermitian(rhobefore),"\ttrace=",np.trace(rhobefore),"\tis_pos_def",is_pos_def(rhobefore))
print("rhoafter hermitian = ",is_hermitian(rhoafter),"\ttrace=",np.trace(rhoafter),"\tis_pos_def",is_pos_def(rhoafter))
##try:    
##    validate_density_matrix(rhoafter,qid_shape=(2,2,))
##except ValueError as e:
##    #  is not positive semidefinite or not hermitian or trace not 1
##    print('Failed validate_density_matrix: ' + str(e))


gf= gatefinder()        
resgate2x2= gf.search(rhobefore,rhoafter,0.00001)

##resgate2x2 =np.array( [[1.00000000e+00+1.77448355e-17j, 9.01055114e-17+1.31939093e-16j],
##              [1.11022302e-16+0.00000000e+00j, 1.00000000e+00+0.00000000e+00j]], dtype=np.complex128)
 
print("seed",seed,"idxrho",idxrho)
print("resgate2x2\n",resgate2x2)
testgate2x2(resgate2x2,rhobefore,rhoafter)



