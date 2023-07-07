import numpy as np
from qiskit.quantum_info import Statevector,random_statevector,DensityMatrix,partial_trace
from qiskit import QuantumCircuit
from clrprint import *
from itertools import combinations
from cirq import is_unitary

               
def tracenumba(rho,nq):
    #traceq=[[0,10,1,11, 4,14,5,15],[0,5,2,7,8,13,10,15]]
    rhoqbit= np.zeros(4,dtype=np.complex128)
    rhof= rho.flatten()
    if nq ==0:
        inc = 10
        idx=[0,1,4,5]
    else:
        inc = 5
        idx=[0,2,8,10]            
    for i in range(4):
        x=idx[i]        
        rhoqbit[i]=rhof[x] + rhof[x+inc]
        #print(i," = (rho[",x,"] + rho [",x+inc,"])")
    return rhoqbit.reshape([2,2])


def getqbitstotraceout(nq,qbitsin):
    qbitsout=[]
    for i in range(nq):
        if not i in qbitsin:                    
            qbitsout.append(i)
        
    return qbitsout

def mytraceqiskit(nq,sv_or_dm,qbitsin):
    
    qbitsout= getqbitstotraceout(nq,qbitsin)
    d =partial_trace(sv_or_dm,qbitsout)    
    return d

def makedms(nq,sv,c2):        
    dms=[]
    for i in range(len(c2)):
        rho=mytraceqiskit(nq,sv,c2[i])
        dms.append(rho)
    return dms

def normalcx(nq,sv,ct,c2):
    qc= QuantumCircuit(nq)
    qc.cx(ct[0],ct[1])
    dmsbefore = makedms(nq,sv,c2)
    svafter=sv.evolve(qc)
    dmsafter=makedms(nq,svafter,c2)
    return dmsbefore,dmsafter,svafter

def rho2x2_cx(sv_or_density,ct):
    qc= QuantumCircuit(2)
    qc.cx(ct[0],ct[1])
    return sv_or_density.evolve(qc)    
    
        
def dmcx(nq,sv,ct,c2):
    qc=QuantumCircuit(2)
    qc.cx(0,1)
    dms=makedms(nq,sv,c2)
    CX_on = -1
    for i in range(len(dms)):
        if c2[i]==ct:                        
            dmsnew= dms[i].evolve(qc)
            dif=sum(np.abs(dms[i].data.flatten()-dmsnew.data.flatten()))
                                    
            if dif > 1e-14:
                print("CX ONNNN",ct,"dif",dif)
                dms[i]=dmsnew
                CX_on=i
            else:
                print("CX OFFFFF",ct,"dif",dif)            
            break
                
##    if CX_on > -1:        
##        target = ct[1]
##        for i in range(len(dms)):
##            if i != CX_on:                
##                if c2[i][0]==target:
##                    print("XXXXXXXXX",c2[i]," target first",target)
##                    qc=QuantumCircuit(2)
##                    qc.x(0)                    
##                    
##                    dms[i]= dms[i].evolve(qc)
##                elif c2[i][1] == target:
##                    print("XXXXXXXXX",c2[i]," target second",target)
##                    qc=QuantumCircuit(2)
##                    qc.x(1)
##                    dms[i]= dms[i].evolve(qc)
##                    
            
    return dms
def getrhoIndexchangePC(dm1,dm2,idxrho,idxcasilla,bPrint=True):
    f1=dm1[idxrho].data.flatten()
    f2=dm2[idxrho].data.flatten()
    res=f2[idxcasilla]/f1[idxcasilla]
    if bPrint:
        print("rho",idxrho,"casilla",idxcasilla,"percent changed",round(res,5))
    return res
    
def compare_dms_dms(dm1,dm2,c2,stitle="", color='blue'):
    print("\ncomparing DM objects",stitle)
    traceq=[[0,10,1,11, 4,14,5,15],[0,5,2,7,8,13,10,15]]
    
    nch=0
    nqch=0
    
    for n in range(len(dm1)):
        f1=dm1[n].data.flatten()
        f2=dm2[n].data.flatten()
        dmequals=np.abs(f1-f2)        
        lc0=[]
        lc1=[]            
        lcmix=[]
        numchanged=0
        dif=sum(dmequals)
        for i in range(len(dmequals)):                
            if dmequals[i] > 1e-15:
                numchanged+=1
                if i in traceq[0]:                                                
                    lc0.append(i)                                                                            
                elif i in traceq[1]:
                    lc1.append(i)
                else:
                    lcmix.append(i)
                                    
                    
        if numchanged == 0:
            clrprint("\n\t",n,") compare dm", c2[n]," IGUALES",clr='blue')
            clrprint("\tnorm1",np.linalg.norm(f1),"\tnorm2",np.linalg.norm(f2))
            continue

        clrprint("\n ",n,") compare dm", c2[n],"dif",dif," changed:",numchanged,                     
                 "\n\tQ0",lc0,"\n\tQ1",lc1,"\n\tQmix",lcmix, clr=color)
        clrprint("\tnorm1",np.linalg.norm(f1),"\tnorm2",np.linalg.norm(f2))
#            nqch+=self.comparequbits(2,dm1,dm2, color=color)            
        nch+=1
    if nch==0:
        clrprint(" No changes in DMs 4x4")
##        if nqch==0:
##            clrprint("\tNo changes in rhos 2x2 qbits")
        

    
def savefileBPNsvtorho(sfile,nq,ct,numpats,semilla,c2):
    
    nstates=int(2**nq)
    qc= QuantumCircuit(nq)    
    qc.cx(ct[0],ct[1])
    
    #listin=[0,5,2,7,8,13,10,15]
    listin=[0,5,8,13,10,15]  #2, 7 conjugates of 8,13   
    #listout=[1, 4, 11, 14,3, 6, 9, 12]
    #listout=[ 4, 11, 6,12] #1,14,3,9 conjugates for 4,11,12,6
    listout=[4]
    
    numin=len(listin)    
    numout=len(listout)
    
    with open(sfile, 'wt') as f:
        f.write("PATRONES {:d}\n".format(int(numpats)))                                    
        f.write("INPUTS {:d}\n".format(int(numin*2)))                          
        f.write("OUTPUTS {:d}\n".format(int(numout*2)))                          
        for i in range(numpats):
            
            sv = random_statevector(nstates,i+semilla)            
            svafter=sv.evolve(qc)
            normalbefore, normalafter,svafter = normalcx(nq,svafter,ct,c2)


            rhoin=normalafter[1].data.flatten()
            for k in range(numin):                                
                f.write("{:.8f}\n".format(rhoin[listin[k]].real))
                f.write("{:.8f}\n".format(rhoin[listin[k]].imag))

            rhoout=normalafter[1].data.flatten()
            
            for k in range(numout):                                
                f.write("{:.8f}\n".format(rhoout[listout[k]].real))
                f.write("{:.8f}\n".format(rhoout[listout[k]].imag))
                        
        
        f.close()
        print("FILE SAVED")


def getdiferentes(a,b):
    l=[]
    for i in range(len(a)):
        d = abs(a[i]-b[i])
        if d > 1e-13:
            l.append(i)                        
    return l
def array1QvaluesFromRHO2(q_0_1, rho2):
    tr=[[0,10,1,11, 4,14,5,15],[0,5,2,7,8,13,10,15]]
    qtr=tr[q_0_1]
    arr= np.zeros(len(qtr),dtype = np.complex128)
    for i in range(len(qtr)):
        arr[i]=rho2 [qtr[i]]
    return arr

def printRHO(rho,listidx):
    for i in range(len(listidx)):
        x = listidx[i]
        print(x,")",round(rho[x],6))
        
def CompareRHOS(rhoA,rhoB,lindexs):
    print("\n CompareRHOS")
    for i in range(len(lindexs)):
        x = lindexs[i]        
        print(x,"dif",round(abs(rhoA[x]-rhoB[x]),5))
                
        
def TransvaseDM_RHO_Q(dm,irhofrom,irhoto,qfrom,qto):
    tr=[[0,10,1,11, 4,14,5,15],[0,5,2,7,8,13,10,15]]    
    a = dm[irhofrom].data.flatten()
    b = dm[irhoto].data.flatten()
    tra=tr[qfrom]
    trb=tr[qto]
    otr=tr[int(not qfrom)]
    for k in range(len(tr[0])):
        if tra[k] not in otr:
            b[trb[k]]=a[tra[k]]

    dm[irhoto]=DensityMatrix(b.reshape([4,4]))
            
        
        
        
def makeListcombinations(nq,nqtr=2):        
        c2=[]
        lstq= np.arange(nq)#.tolist()
        c2=list(combinations(lstq, nqtr))
        for i in range(len(c2)):
            c2[i]= list(c2[i])
        return c2
def testfinderhos(seed, idxrho):
    nq=5
    ct=[0,1]
    c2=makeListcombinations(nq,len(ct))    
    sv = random_statevector(int(2**nq),seed)    
    normalbefore, normalafter,svafter = normalcx(nq,sv,ct,c2)    
    rhoBefore=np.copy(normalbefore[idxrho].data)
    rhoAfter=np.copy(normalafter[idxrho].data)
    return rhoBefore,rhoAfter
def testmain(seed):
    nq=5
    ct=[0,1]
    c2=makeListcombinations(nq,len(ct))    
##    savefileBPNsvtorho('BPNtestcx01.txt',nq,ct,150,1000,c2)
##    exit(0)
    sv = random_statevector(int(2**nq),seed)    
    normalbefore, normalafter,svafter = normalcx(nq,sv,ct,c2)    
    myafter=dmcx(nq,sv,ct,c2)    
    
    #compare_dms_dms(normalbefore,normalafter,c2,"state before with state after cx", color="red")
    compare_dms_dms(normalafter,myafter,c2,"state after with dm after", color="red")

##    rhoA=np.copy(normalbefore[1].data)
##    rhoB=np.copy(normalafter[1].data)
##    print("normalbefore[1]\n",np.round(rhoA,15))
##    print("normalafter[1]\n",np.round(rhoB,15))
    
##    lbad=[1,4,11,14,3,6,9,12] # 1 = conj(4) y 14 = conj(11) y 3= conj(12) y 9 conj(6)
##    lok=[0,5,2,7,8,13,10,15] #2=conj(8) y 7 = conj(13)
##    
##    printRHO(rhoA,lbad)    
    #printRHO(rhoB,lok)
    #CompareRHOS(rhoA.flatten(),rhoB.flatten(),np.arange(16))

testmain(seed=10)


#TransvaseDM_RHO_Q(dm=myafter,irhofrom=0,irhoto=1,qfrom=0,qto=0)
#compare_dms_dms(normalafter,myafter,c2,"state after with dm after", color="red")




        
        
