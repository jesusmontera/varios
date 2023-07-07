import numpy as np
def IdentityGate():
    gate=np.zeros([2,2],dtype=np.complex128)    
    gate [0][0]=  1
    gate [1][1]=  1    
    return gate

def cxgate(control):
    if control==0:
        gate = [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]]
    else:
        gate = [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]]
        
    return np.array(gate,dtype=np.complex128)
def HadamardGate1Qbit():
    h=np.zeros([2,2],dtype=np.complex128)
    s2=np.sqrt(2)
    h [0][0]=  1 / s2
    h [0][1]=  1 / s2
    h [1][0]=  1 / s2
    h [1][1]=  -1 / s2
    return h
    
def HadamardGate(nq):
    h=HadamardGate1Qbit()    
    if nq==0:
        h = np.kron(IdentityGate(), h)
    else:
        h = np.kron(h,IdentityGate())        
    return h

def XGate(nq):
    x=np.zeros([2,2],dtype=np.complex128)    
    x [0][1]=  1
    x [1][0]=  1
    if nq==0:
        x = np.kron(IdentityGate(), x)
    else:
        x = np.kron(x , IdentityGate())        
    return x
