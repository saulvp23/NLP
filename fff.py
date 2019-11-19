# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:00:30 2019

@author: saul.villamizar
"""
import numpy as np

def split_half(a):
    half = len(a)//2
    return a[:half], a[half:]

def f (X):
    _X = np.copy(X)
    x, y = split_half(_X)
    global L
    global nb
    global _y0,_ylast
    F = []
    for i in range(nj+1):
        if i==0:
            F.append(L[i]*(y[i]+_y0)/2)
        elif i == (nj):
            F.append(L[i]*(_ylast+y[i-1])/2)
        else:
            F.append(L[i]*(y[i]+y[i-1])/2)
    return sum(F)

def g (X):
    global L
    global nb
    _X = np.copy(X)
    Gx, Gy = split_half(_X)
    Gx.fill(0)
    for i in range(nj):
        Gy[i] = L[i]/2+L[i+1]/2
    G = np.concatenate((Gx, Gy), axis=None)
    return G

def h (X):
    global nb,L
    global _x0,_y0
    _X = np.copy(X)
    x, y = split_half(_X)
    nodes = []
    for i in range(nb+1):
        if i==0:
            nodes.append([_x0,_y0])
        elif i==nb:
            nodes.append([_xlast,_ylast])
        else:
            nodes.append([x[i-1],y[i-1]])
    nodes = np.array( nodes )
    li = np.array([np.linalg.norm(nodes[i] - nodes[i-1]) for i in range(1,nb+1)])
    H = li**2 - L**2
    return H 

def j (X):
    global nj,nb
    _X = np.copy(X)
    x, y = split_half(_X)
    ghx = []
    for jnb in range (nb):
        ghx.append([])
        for inj in range (nj):
            if inj == 0 and jnb==0:
                ghx[jnb].append(2*(x[inj]-_x0)*(jnb==0))      
            elif inj == nj-1 and jnb == nb-1:
                ghx[jnb].append(2*(_xlast-x[inj])*(jnb==(nb-1)))
            elif jnb<nj :
                ghx[jnb].append(2*(x[jnb]-x[jnb-1])*(inj==jnb) -2*(x[jnb]-x[jnb-1])*(inj==(jnb-1))) 
            else :
                ghx[jnb].append(2*(x[inj]-x[inj-1])*(inj==jnb) -2*(x[inj]-x[inj-1])*(inj==(jnb-1))) 
                
    
    ghx = np.array([ghx[jnb] for jnb in range(nb)])
    ghy = []
    for jnb in range (nb):
        ghy.append([])
        for inj in range (nj):
            if inj == 0 and jnb==0:
                ghy[jnb].append(2*(y[inj]-_y0)*(jnb==0))      
            elif inj == nj-1 and jnb == nb-1:
                ghy[jnb].append(2*(_ylast-y[inj])*(jnb==(nb-1)))
            elif jnb<nj :
                ghy[jnb].append(2*(y[jnb]-y[jnb-1])*(inj==jnb) -2*(y[jnb]-y[jnb-1])*(inj==(jnb-1))) 
            else :
                ghy[jnb].append(2*(y[inj]-y[inj-1])*(inj==jnb) -2*(y[inj]-y[inj-1])*(inj==(jnb-1))) 
            
    ghy = np.array([ghy[jnb] for jnb in range(nb)])
    
    J = np.array([np.concatenate((ghx[i], ghy[i]), axis=None) for i in range(0,nb)])
    return J



def phi( _X, _Lamda, _C):
    H,J = oracle.hj(_X, mode=1)
    F,G = oracle.fg(_X, mode=1)
    L = F + np.dot(_Lamda.T[0],H)+_C/2*np.linalg.norm(H)**2
    return L

def gphi( _X, _Lamda, _C):
    H,J = oracle.hj(_X, mode=2)
    F,G = oracle.fg(_X, mode=2)
    GL = G+np.dot(_Lamda.T[0]+_C*H,J)
    return GL

def l( _X, Lamda, C):
    H,J = oracle.hj(_X, mode=1)
    F,G = oracle.fg(_X, mode=1)
    L = F + np.dot(Lamda.T,H)+C/2*np.linalg.norm(H)**2
    return L

def gl( _X, Lamda, C):
    H,J = oracle.hj(_X, mode=2)
    F,G = oracle.fg(_X, mode=2)
    GL = np.concatenate((G+np.dot(Lamda.T+C*H,J), H), axis=None)
    return GL

class oracle:
        
    def fg(X, mode):
        if mode==1:
            return f(X),None 
        elif mode==2:
            return f(X),g(X)
        elif mode==3:
            return None,g(X)
        else:
            print('Not on the list')

    def hj(X, mode):
        if mode==1:
            return h(X),None 
        elif mode==2:
            return h(X),j(X)
        elif mode==3:
            return None,j(X)
        else:
            print('Not on the list')

    def lgl(X,Lamda,C, mode):
        if mode==1:
            return l( X, Lamda, C),None 
        elif mode==2:
            return l(X, Lamda, C),gl( X, Lamda, C)
        elif mode==3:
            return None,gl(X, Lamda, C, oracle)
        else:
            print('Not on the list')
            
    def phig(X,Lamda,C, mode):
        if mode==1:
            return phi(X, Lamda, C),None 
        elif mode==2:
            return phi(X, Lamda, C),gphi( X, Lamda, C)
        elif mode==3:
            return None,gphi(X, Lamda, C, oracle)
        else:
            print('Not on the list')
    

def Armijo(X,d,Lamda,C,MaxItLineSearch,t0 = 100, theta=0.2, m=0.001):
    p=0
    t = [t0]
    while True:
        _phi, _gphi = oracle.phig(X,Lamda,C, mode=2)
        _phi1, _gphi1 = oracle.phig(X+t[-1]*d,Lamda,C, mode=1)
        alfa= m*t[-1]*np.dot(_gphi,d)
        if _phi1 <= _phi + alfa:
            return t[-1]
        else:
            t.append(theta*t[-1])
            if p<MaxItLineSearch:
                p+=1
            else:
                print(f'Max iterations \n\t direction: {d} \n\t step:{t[-1]}')
                raise ValueError('Max iteration')

            
def algorithme2(x0,Lamda,C, MaxIt=100,tol = 10e-6, MaxItLineSearch=50, stepkk = 100):
    kk=0
    Xkk =  [x0] 
    while True:
        f, g = oracle.phig(Xkk[-1],Lamda,C, mode=2)
        if np.linalg.norm(g)<tol:
            print (f'Converged on {X[-1]}')
            return Xkk[-1], Xkk
        else:
            d = -g
            try:
                stepkk = Armijo(Xkk[-1],d,Lamda,C, MaxItLineSearch=MaxItLineSearch, t0 = 1)
            except ValueError:
                print('Max armijo line search iterations')
                return None,Xkk                                
            Xkk.append(Xkk[-1] + (stepkk)*d)
            if kk<MaxIt:
                kk+=1
            else:
                print('Max iterations algo2')
                return None,Xkk
            
def algorithm(x0,lamda0,c0, tol = 10e-6, MaxIt=1000, stepC=0.5):
    k=0
    C = [c0]
    Lamda = [lamda0]
    Xk =  [x0] 
    Xkh = []
    while True:
        Xk.append(0)
        Xkh.append([])
        Xk[-1],Xkh[-1] = algorithme2(Xk[-2],Lamda[-1],C[-1])
        lagrange, gradient = oracle.lgl(Xk[-1],Lamda[-1],C[-1], mode = 2)
        if np.linalg.norm(gradient)<tol:
            print (f'Converged on {X[-1]}')
            return X[-1],X,Lamda[-1],Lamda,C[-1],C
        else:
            hk,jk = oracle.hj(X[-1], mode=1)
            Lamda.append(Lamda[-1]+C[-1]*hk)
            C = C+stepC
            if k<MaxIt:
                k+=1
            else:
                print('Max iterations algo1')
                return None,X,Lamda,C

nb = 3
nj = nb - 1
L = 5*np.ones(nb)
lamda = np.array([ [0] for i in range(nb)])
c=1
_x0=0
_xlast=11

_y0=0
_ylast=0

_x = np.array([2,9])
_y = np.array([-5,-3])
x = np.concatenate((_x,_y),axis=None)
#X,L,C = algorithm(x,lamda,c)
a,b = algorithme2(x,lamda,c)

