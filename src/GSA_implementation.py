# -*- coding: utf-8 -*-
"""
Python code of Feature Selection using Gravitational Search Algorithm (GSA) and Support Vector Machine (SVM)


Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

Purpose: Main file of Gravitational Search Algorithm(GSA) 
            for minimizing of the Objective Function

Code compatible:
 -- Python: 2.* or 3.*
"""

import random
import numpy as np
import math
from solution import solution
import time

#move.py
def move(PopSize,dim,pos,vel,acc):
    for i in range(0,PopSize):
        for j in range (0,dim):
            r1=random.random()
            vel[i,j]=r1*vel[i,j]+acc[i,j]
            pos[i,j]=pos[i,j]+vel[i,j]
    
    return pos, vel

#massCalculation.py
def massCalculation(fit,PopSize,M):
    Fmax = max(fit)
    Fmin = min(fit)
    Fsum = sum(fit)        
    Fmean = Fsum/len(fit)
        
    if Fmax == Fmin:
        M = np.ones(PopSize)
    else:
        best = Fmin
        worst = Fmax
        
        for p in range(0,PopSize):
           M[p] = (fit[p]-worst)/(best-worst)
            
    Msum=sum(M)
    for q in range(0,PopSize):
        M[q] = M[q]/Msum
            
    return M

#gfield.py
def gField(PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower):
    final_per = 2
    if ElitistCheck == 1:
        kbest = final_per + (1-l/iters)*(100-final_per)
        kbest = round(PopSize*kbest/100)
    else:
        kbest = PopSize
            
    kbest = int(kbest)
    ds = sorted(range(len(M)), key=lambda k: M[k],reverse=True)
        
    Force = np.zeros((PopSize,dim))
    # Force = Force.astype(int)
    
    for r in range(0,PopSize):
        for ii in range(0,kbest):
            z = ds[ii]
            R = 0
            if z != r:                    
                x=pos[r,:]
                y=pos[z,:]
                esum=0
                imval = 0
                for t in range(0,dim):
                    imval = ((x[t] - y[t])** 2)
                    esum = esum + imval
                    
                R = math.sqrt(esum)
                
                for k in range(0,dim):
                    randnum=random.random()
                    Force[r,k] = Force[r,k]+randnum*(M[z])*((pos[z,k]-pos[r,k])/(R**Rpower+np.finfo(float).eps))
                    
    acc = np.zeros((PopSize,dim))
    for x in range(0,PopSize):
        for y in range (0,dim):
            acc[x,y]=Force[x,y]*G
    return acc

#gconstant.py
def gConstant(l,iters):
    alfa = 20
    G0 = 10
    Gimd = np.exp(-alfa*float(l)/iters)
    G = G0*Gimd
    return G



        
def GSA(objf,lb,ub,dim,PopSize,iters,df):
    
    # Convert lb and ub to numpy arrays
    lb = np.array(lb)
    ub = np.array(ub)
    
    
    # GSA parameters
    ElitistCheck =1
    Rpower = 1 
     
    s=solution()
        
    """ Initializations """
    "numpy.zeros(x) digunakan untuk menghasilkan 1 dimensional array dengan angka nol"
    "numpy.zeros(x,y) digunakan untuk menghasilkan 2 dimensional array dengan angka nol (x baris, y kolom)"
    
    vel=np.zeros((PopSize,dim))
    fit = np.zeros(PopSize)
    M = np.zeros(PopSize)
    gBest=np.zeros(dim)
    gBestScore=float("inf")#One can use float("inf") as an integer to represent it as infinity
    
    """ Generate Search Space """
    " syntax di bawah digunakan untuk menggenerate angka random dari range 0 s/d 1 dgn bentuk array 5 baris (popsize) x 4 kolom (dim)"
    
    pos=np.random.uniform(0,1,(PopSize,dim)) *(ub-lb)+lb
    
    convergence_curve=np.zeros(iters)
    
    print("GSA is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(0,iters): #looping dari l = 0 selama dalam range 0 sampai 10
        for i in range(0,PopSize):
            l1 = [None] * dim #bikin matriks [None] * 4
            l1=np.clip(pos[i,:], lb, ub)
            pos[i,:]=l1

            #Calculate objective function for each particle
            fitness=[]
            fitness=-objf(l1,df)
            fit[i]=fitness
    
                
            if(gBestScore>fitness):
                gBestScore=fitness
                gBest=l1
        
        """ Calculating Mass """
        M = massCalculation(fit,PopSize,M)

        """ Calculating Gravitational Constant """        
        G = gConstant(l,iters)        
        
        """ Calculating Gfield """        
        acc = gField(PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower)
        
        """ Calculating Position """        
        pos, vel = move(PopSize,dim,pos,vel,acc)
        
        convergence_curve[l]=gBestScore
      
        if (l%1==0):
                # Just for display, added the negative sign.
               print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(-(gBestScore))]);
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.gBest=gBest
    s.best_fitness = gBestScore  # Set best_fitness (10/11/24)
    s.best_solution = gBest      # Set best_solution (10/11/24)
    s.Algorithm="GSA"
    s.objectivefunc=objf.__name__

    return s
         
    
