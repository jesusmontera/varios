import numpy as np
import cirq
from qiskit import QuantumCircuit,transpile
from qiskit.circuit.random import random_circuit
import matplotlib.pyplot as plt
import re
#from qiskit.circuit import Instruction
import qiskit.quantum_info as qi
from cirq import bloch_vector_from_state_vector
from mymct import mymct
from mymctcirq import mymctcirq

import random


def printCCXtext(s):
    a=s[0]
    b=s[1]
    x=s[2]
    print("    ## CCX translated begin")    
    print("    qc.h (",x," )")
    print("    qc.cx (",b," , ",x," )")
    print("    qc.rz(-np.pi/4 , ",x," )")
    print("    qc.cx (",a," , ",x," )")
    print("    qc.rz(np.pi/4 , ",x," )")
    print("    qc.cx (",b," , ",x," )")
    print("    qc.rz(np.pi/4 , ",b," )")
    print("    qc.rz(-np.pi/4 , ",x," )")
    print("    qc.cx (",a," , ",x," )")
    print("    qc.cx (",a," , ",b," )")
    print("    qc.rz(np.pi/4 , ",a," )")
    print("    qc.rz(-np.pi/4 , ",b," )")
    print("    qc.cx (",a," , ",b," )")
    print("    qc.rz(np.pi/4 , ",x," )")
    print("    qc.h (",x," )")

    print("    ## CCX translated end")

def diferenciaBloches(sv1,sv2,nq,nbloches):        
    dif =0    
        
    for i in range(nbloches):
        b1 = bloch_vector_from_state_vector(sv1,nq-1-i)
        b2 = bloch_vector_from_state_vector(sv2,nq-1-i)            
        dif +=sum(np.abs(b1-b2))
    return dif

def printmctfunction(qc,slabel):        
        sc=""
        n= len(qc.data)
        lst=[]                
        for i in range(n):
            tu=qc.data[i]
            sc=tu[0].qasm()
            if tu[0].params !=  []:
                sc = sc.replace(")"," , ")
                sc = sc.replace("pi","np.pi")
                
            else:
                sc = sc.replace(")","") + " ("
            
            for k in range(len(tu[1])):                
                if k > 0:
                    sc += " , "                    
                
                x=tu[1][k].index                
                sc += slabel[x]
                
                    
            sc +=" )"
            lst.append(sc)
        return lst



def getlabelsfuncparams(nq,target,numfree=0):
    slabel=[]
    x=0
    a=0
    for i in range(nq+numfree):
        if i==target:            
            slabel.append('t')
        else:
            if i >= nq: # auxfreequbits
                sc= "aux["+  str(a) + "]"
                a+=1
                slabel.append(sc)                
            else:
                sc= "c["+  str(x) + "]"
                x+=1
                slabel.append(sc)

    
        
    return slabel

def makemctfunction(ncontrols):
    print("def mymct" + str(ncontrols)+"(qc,c,t):")
    nq= ncontrols+1
    qc=QuantumCircuit(nq)
    c = list(np.arange(0,nq-1,1, dtype=int))
    target = nq-1    
    qc.mct(c,target)
    qct = transpile(qc,basis_gates=['rx','ry','rz','x', 'cx','h'])    
    numins= len(qct.data)
    print(" #nº of instructions ",numins )
    slabel=getlabelsfuncparams(nq,target)
    

    l= printmctfunction(qct,slabel)    
    for i in range(numins):     
        print("    qc."+str(l[i]))
def find_C_T_AUX(nqfind,ci,fi,target):
    if nqfind == target:
        return "t"
    
    for i in range(len(ci)):
        if ci[i] == nqfind:
            return "c[" + str(i) + "]"
    for i in range(len(fi)):
        if fi[i] == nqfind:
            return "aux[" + str(i) + "]"
    
    print("ERROR EN find_C_T_AUX")
    exit(0)

    
def makecirqmct(ncontrols):

    numfree = ncontrols -2
    nq= ncontrols+1    
    target = nq-1
    print("def mymctcirq" + str(ncontrols)+"(qc,c,t,aux):")
    #print("cirq nº qubits",nq,"n controls",ncontrols)
    
    q = cirq.LineQubit.range(nq+numfree)

    slabel=getlabelsfuncparams(nq,target,numfree)
    c=[]
    ci=[]
    for i in range (ncontrols):
        c.append(q[i])
        ci.append(i)
        
    
    qfree=[]
    fi=[]
    for i in range (nq,nq+numfree):
        qfree.append(q[i])
        fi.append(i)
        
        
    
    d = cirq.decompose_multi_controlled_x(c,q[target],qfree)        
    numinstrucs=len(d)
    numccx=0
    numerr=0
    sregname=["","",""]
    for i in range(numinstrucs):
                        

        s = re.split("[, ()π:]+", str(d[i]))        
        #print(s)                
        if s[0]=="TOFFOLI":
            #print("    qc.ccx ( " + slabel[int(s[1])] + " , " + slabel[int(s[2])] + " , " + slabel[int(s[3])]  + ")" )
            for k in range(3):
                sregname[k] = find_C_T_AUX(int( s[k+1]),ci,fi,target)
            printCCXtext(sregname)
            numccx+=1
        elif s[0]=="Ry":
            print("    qc.ry ( " + s[1] + " * np.pi ," + slabel[int(s[2])] + ")" )
        elif s[0]=="Rx":
            print("    qc.rx ( " + s[1] + " * np.pi ," + slabel[int(s[2])] + ")" )
        elif s[0]=="Rz":
            print("    qc.rz ( " + s[1] + " * np.pi ," + slabel[int(s[2])] + ")" )           
        elif s[0]=="CNOT":
            print("    qc.cx ( " + slabel[int(s[1])] + " , " + slabel[int(s[2])]  + ")" )
        else:
            numerr+=1
            
            
    print("errors in conversion ",numerr)
    numinstrucs += numccx*14 # each ccx is 15 len (minus ccx replaced)
    print("cirq mct with ",numfree,"free", "have", numinstrucs,"instrucs (",numccx,"ccx)")
    
def testmymctfunction(nq,ncontrols):
    difsv=0
    difblochs=0
    difsvcirq=0
    difblochscirq=0

    numfree=ncontrols+1 # ideal ncontrols-2
    aux = list(np.arange(nq,nq+numfree,1, dtype=int))
    
    for i in range(30):        
        qc1=random_circuit(num_qubits=nq+numfree, depth=5, max_operands=2)
##        qc1=QuantumCircuit(nq+numfree)
##        qc1.h([0,3,6])
##        qc1.x([0,1,6])
##        qc1.ry(0.5,0)
##        qc1.cx(0,2)
##        qc1.ry(0.9,3)
##        
        qc2=qc1.copy()
        qc3=qc1.copy()    
        r= random.sample(range(0,nq), ncontrols+1)
        c=r[0:ncontrols]
        t = r[ncontrols]
        
        print("executing mct qiskit")        
        qc1.mct(c,t)
        print("executing mymct( mode qiskit)  ERROR BLOCH",difblochs)           
        mymct(qc2,c,t)
        print("executing mymct (mode cirq) ERROR BLOCH",difblochscirq)                
        mymctcirq(qc3,c,t,aux)
        
        sv1=qi.Statevector.from_instruction(qc1).data
        sv2=qi.Statevector.from_instruction(qc2).data        
        difsv += sum(np.abs(sv1-sv2))
        difblochs += diferenciaBloches(sv1,sv2,nq= qc1.num_qubits , nbloches = nq)        
    
        sv3=qi.Statevector.from_instruction(qc3).data
        difsvcirq += sum(np.abs(sv1-sv3))
        difblochscirq += diferenciaBloches(sv1,sv3,nq= qc1.num_qubits , nbloches = nq)
    print("\ndif states  mymct mct qiskit",difsv)
    print("dif bloches mymct mct qiskit",difblochs)
    
    print("\ndif states  mymct cirq mct",difsvcirq)
    print("dif bloches mymct cirq mct",difblochscirq)

    
    

#testmymctfunction(nq=8,ncontrols=5)
#makemctfunction(ncontrols=8)
makecirqmct(ncontrols=20)

        
        
        
        

