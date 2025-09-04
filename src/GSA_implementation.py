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
def massCalculation(fit, PopSize, M):
    Fmax = float(np.max(fit))
    Fmin = float(np.min(fit))

    if Fmax == Fmin:
        M = np.ones(PopSize)
    else:
        best, worst = Fmin, Fmax
        for p in range(PopSize):
            M[p] = (fit[p] - worst) / (best - worst)

    Msum = float(np.sum(M)) + np.finfo(float).eps
    for q in range(PopSize):
        M[q] = M[q] / Msum
    return M


#gfield.py
def gField(PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower):
     # Slightly larger final percentage so force does not collapse to 1 body too early
    final_per = 2

    # if ElitistCheck == 1:
    #     kbest = final_per + (1-l/iters)*(100-final_per)
    #     kbest = round(PopSize*kbest/100)
    # else:
    #     kbest = PopSize
            
    # kbest = int(kbest)


    if ElitistCheck == 1:
        kbest_pct = final_per + (1 - l/iters) * (100 - final_per)
    else:
        kbest_pct = 100.0


    # --- IMPORTANT: never allow kbest < 5 (helps avoid early collapse)
    kbest = max(5, int(round(PopSize * kbest_pct / 100.0)))

    # Optional tiny debug
    DEBUG_KBEST = True
    if DEBUG_KBEST and l in (0, 1, 2, iters - 2, iters - 1):
        print(f"[iter {int(l)+1}] kbest={kbest}")



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
# def gConstant(l,iters):
#     alfa = 20
#     G0 = 10
#     Gimd = np.exp(-alfa*float(l)/iters)
#     G = G0*Gimd
#     return G

def gConstant(l, iters, G0=10.0, alpha=5.0, G_min_ratio=0.10):
    # slower decay than alpha=20 and never below a small floor
    G = G0 * math.exp(-alpha * l / float(iters))
    return max(G, G0 * G_min_ratio)


        
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

    # WOA-style exploration knobs (local to this run)
    # trial 1
    # p0 = 0.25                 # starting peer-jump rate - will decay over time
    # A_low, A_high = -1.6, 1.6 # narrower A to avoid wild jumps
    # elite_frac = 0.30         # top 30% masses are protected from peer jumps
    # stagnation_patience = 5
    # repulse_frac = 0.25

    p0 = 0.30                 # a bit more peer exploration early
    A_low, A_high = -1.8, 1.8 # slightly broader jumps, still bounded
    elite_frac = 0.20         # protect fewer agents to keep diversity
    stagnation_patience = 4   # respond to plateaus a touch sooner
    repulse_frac = 0.30       # push a few more agents on a burst


    # stagnation tracking
    no_improve = 0
    last_best = gBestScore

    
    """ Generate Search Space """
    " syntax di bawah digunakan untuk menggenerate angka random dari range 0 s/d 1 dgn bentuk array 5 baris (popsize) x 4 kolom (dim)"
    
    pos=np.random.uniform(0,1,(PopSize,dim)) *(ub-lb)+lb
    
    convergence_curve=np.zeros(iters)
    
    print("GSA is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(0,iters): #looping dari l = 0 selama dalam range 0 sampai 10
        for i in range(0,PopSize):
            # l1 = [None] * dim #bikin matriks [None] * 4
            # l1=np.clip(pos[i,:], lb, ub)
            # pos[i,:]=l1

            # #Calculate objective function for each particle
            # fitness=[]
            # fitness=-objf(l1,df)
            # fit[i]=fitness

            l1 = np.clip(pos[i, :], lb, ub)
            pos[i, :] = l1

            # fitness = -objf(l1, df)

            fitness = objf(l1, df)
            fit[i] = fitness

                
            if(gBestScore>fitness):
                gBestScore=fitness
                gBest=l1
        
        """ Calculating Mass """
        M = massCalculation(fit,PopSize,M)

        """ Calculating Gravitational Constant """        
        G = gConstant(l,iters)  
        if no_improve >= stagnation_patience:
            G = max(G, 0.2 * 10.0)         # bump G to at least 20% of G0 for this iter      
        

        # Optional tiny debug
        DEBUG_G = True
        if DEBUG_G and l in (0, 1, 2, iters - 2, iters - 1):
            print(f"[iter {l+1}] G={G:.4f}")



        """ Calculating Gfield """        
        acc = gField(PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower)
        
        """ Calculating Position """        
        # pos, vel = move(PopSize,dim,pos,vel,acc)

        # --- Proposed position from standard GSA update
        pos_prop, vel_prop = move(PopSize, dim, pos.copy(), vel.copy(), acc)

        # # --- WOA-inspired peer exploration (random peer guidance)
        # for i in range(PopSize):
        #     if random.random() < p_peer:
        #         j = random.randrange(PopSize)
        #         while j == i:
        #             j = random.randrange(PopSize)
        #         A = random.uniform(A_low, A_high)
        #         if abs(A) > 1.0:
        #             Xrand = pos_prop[j, :]
        #             Xi    = pos_prop[i, :]
        #             Xi_new = Xrand - A * np.abs(Xrand - Xi)
        #             pos_prop[i, :] = np.clip(Xi_new, lb, ub)


        # --- WOA-inspired peer exploration - gated and decayed
        # protect top-mass elites from random peer jumps
        topk = max(1, int(elite_frac * PopSize))
        elite_idx = set(np.argsort(-M)[:topk])  # indices of largest masses

        # # decay the peer jump probability across iterations
        # p_peer_eff = max(0.02, p0 * (1.0 - l / float(iters)))

        if l < iters * 0.5:
            p_peer_eff = p0
        else:
            p_peer_eff = max(0.05, p0 * (1.0 - (l - iters * 0.5) / (iters * 0.5)))

            
        for i in range(PopSize):
            if i in elite_idx:
                continue
            if random.random() < p_peer_eff:
                j = random.randrange(PopSize)
                while j == i:
                    j = random.randrange(PopSize)

                A = random.uniform(A_low, A_high)
                if abs(A) > 1.0:  # exploration-only regime
                    Xrand = pos_prop[j, :]
                    Xi    = pos_prop[i, :]
                    Xi_new = Xrand - A * np.abs(Xrand - Xi)
                    pos_prop[i, :] = np.clip(Xi_new, lb, ub)


        # --- Small jitter on a fraction of non-elites (to cross thresholds)
        jitter_non_elite_frac = 0.30   # tweakable
        jitter_dim_frac = 0.10         # tweakable
        jitter_sigma = 0.06            # tweakable

        non_elites = [idx for idx in range(PopSize) if idx not in elite_idx]
        if non_elites:
            jitter_count = max(1, int(jitter_non_elite_frac * len(non_elites)))
            jitter_agents = random.sample(non_elites, jitter_count)

            jitter_dims = max(1, int(jitter_dim_frac * dim))
            for i in jitter_agents:
                cols = np.random.choice(dim, size=jitter_dims, replace=False)
                noise = np.random.normal(
                    loc=0.0,
                    scale=jitter_sigma * (ub[cols] - lb[cols]),
                    size=jitter_dims
                )
                pos_prop[i, cols] = np.clip(pos_prop[i, cols] + noise, lb[cols], ub[cols])

        # --- Light bit-flip kick on a few elites (helps escape late plateaus)
        elite_flip_frac = 0.10   # top ~10% by mass
        max_bits_to_flip = 2

        flip_k = max(1, int(elite_flip_frac * PopSize))
        flip_idx = np.argsort(-M)[:flip_k]
        for i in flip_idx:
            nflip = random.randint(1, max_bits_to_flip)
            cols = np.random.choice(dim, size=nflip, replace=False)
            for c in cols:
                # reflect around interval midpoint to "jump the fence"
                mid = 0.5 * (lb[c] + ub[c])
                if pos_prop[i, c] >= mid:
                    pos_prop[i, c] = lb[c] + 0.05 * (ub[c] - lb[c])
                else:
                    pos_prop[i, c] = ub[c] - 0.05 * (ub[c] - lb[c])
            pos_prop[i, :] = np.clip(pos_prop[i, :], lb, ub)

        # --- Optional repulsion burst on stagnation
        if no_improve >= stagnation_patience:


             # push a fraction of agents away from current best using a random peer 
            num_repulse = max(1, int(repulse_frac * PopSize))
            idxs = random.sample(range(PopSize), num_repulse)
            for i in idxs:
                j = random.randrange(PopSize)
                Xrand = pos_prop[j, :]
                Xi    = pos_prop[i, :]
                A = random.uniform(1.2, 2.0)   # stronger push
                Xi_new = Xrand - A * np.abs(gBest - Xi)
                pos_prop[i, :] = np.clip(Xi_new, lb, ub)
            
            # light OBL refresh on ~10 percent of non-elites
            cand = [i for i in range(PopSize) if i not in elite_idx]
            if cand:
                r = max(1, int(0.1 * len(cand)))
                pick = random.sample(cand, r)
                for i in pick:
                    x = pos_prop[i, :]
                    x_opp = lb + ub - x
                    eta = random.uniform(0.3, 0.7)         # sample around the middle
                    mix = eta * x + (1.0 - eta) * x_opp
                    pos_prop[i, :] = np.clip(mix, lb, ub)

            
            no_improve = 0  # reset after burst

        # --- Commit the update
        pos, vel = pos_prop, vel_prop
        
        convergence_curve[l]=gBestScore
      
        if (l%1==0):
                # Just for display, added the negative sign.
            #    print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(-(gBestScore))]);
                 print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(gBestScore)])
        if gBestScore < last_best:   # recall: smaller is better in your code
            no_improve = 0
            last_best = gBestScore
        else:
            no_improve += 1

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
         
    
