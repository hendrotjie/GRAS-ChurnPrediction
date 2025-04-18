# -*- coding: utf-8 -*-
"""
Python code of Feature Selection using Gravitational Search Algorithm (GSA) and Support Vector Machine (SVM)


Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Defining the solution class
 
Code compatible:
 -- Python: 2.* or 3.*

"""

class solution:
    def __init__(self):
        self.best_fitness = None  # Initialize best_fitness to store the best fitness value
        self.best_solution = []  # Add this if you want to store the best solution (features selected)
        self.convergence = []
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.gBest=[]
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0
        self.trainAcc=None
        self.trainTP=None
        self.trainFN=None
        self.trainFP=None
        self.trainTN=None
        self.testAcc=None
        self.testTP=None
        self.testFN=None
        self.testFP=None
        self.testTN=None
