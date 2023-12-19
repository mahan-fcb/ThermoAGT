#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from Bio.PDB import *
import numpy as np
import math
from glob import glob
import pandas as pd
import pickle
import MDAnalysis as mda
#from MDAnalysis.tests.datafiles import GRO, XTC
from MDAnalysis.analysis import dihedrals
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from Bio.PDB import *
import numpy as np
import math
from glob import glob
import pandas as pd
import pickle
import os

def two_maps(res_list):
    
    coords = []
    S = 0
    
    for res in res_list:
        if str(res)[8:12] == 'HOH':
            break
        else:
            for atoms in res:
                if str(atoms) == '<Atom CA>':
                    
                    coords.append(atoms.get_coord())
                    S +=1

    coords = np.array(coords)
    map_of_cont = np.zeros((len(coords), len(coords)))
    map_of_dist = np.zeros((len(coords), len(coords)))
    
    for i in range(len(coords) - 1):
        for j in range (i + 1, len(coords)):
            
            map_of_dist[i, j] = math.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2 + (coords[i, 2] - coords[j, 2])**2)
            map_of_dist[j, i] = map_of_dist[i, j]
            
            if math.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2 + (coords[i, 2] - coords[j, 2])**2) <= 8:
                
                map_of_cont[i,j] = 1
                map_of_cont[j,i] = 1
                
    return(map_of_dist, map_of_cont)


# In[2]:


def two_mapsN(res_list):
    
    coords = []
    S = 0
    
    for res in res_list:
        if str(res)[8:12] == 'HOH':
            break
        else:
            for atoms in res:
                if str(atoms) == '<Atom N>':
                    
                    coords.append(atoms.get_coord())
                    S +=1

    coords = np.array(coords)
    map_of_cont = np.zeros((len(coords), len(coords)))
    map_of_dist = np.zeros((len(coords), len(coords)))
    
    for i in range(len(coords) - 1):
        for j in range (i + 1, len(coords)):
            
            map_of_dist[i, j] = math.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2 + (coords[i, 2] - coords[j, 2])**2)
            map_of_dist[j, i] = map_of_dist[i, j]
                
    return(map_of_dist)


# In[3]:


def two_mapsC(res_list):
    
    coords = []
    S = 0
    
    for res in res_list:
        if str(res)[8:12] == 'HOH':
            break
        else:
            for atoms in res:
                if str(atoms) == '<Atom C>':
                    
                    coords.append(atoms.get_coord())
                    S +=1

    coords = np.array(coords)
    map_of_cont = np.zeros((len(coords), len(coords)))
    map_of_dist = np.zeros((len(coords), len(coords)))
    
    for i in range(len(coords) - 1):
        for j in range (i + 1, len(coords)):
            
            map_of_dist[i, j] = math.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2 + (coords[i, 2] - coords[j, 2])**2)
            map_of_dist[j, i] = map_of_dist[i, j]
                
    return(map_of_dist)


# In[4]:


def two_mapsCB(res_list):
    
    coords = []
    S = 0
    
    for res in res_list:
        if str(res)[8:12] == 'HOH':
            break
        else:
            for atoms in res:
                if str(atoms) == '<Atom CB>':
                    
                    coords.append(atoms.get_coord())
                    S +=1

    coords = np.array(coords)
    map_of_cont = np.zeros((len(coords), len(coords)))
    map_of_dist = np.zeros((len(coords), len(coords)))
    
    for i in range(len(coords) - 1):
        for j in range (i + 1, len(coords)):
            
            map_of_dist[i, j] = math.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2 + (coords[i, 2] - coords[j, 2])**2)
            map_of_dist[j, i] = map_of_dist[i, j]
                
    return(map_of_dist)


# In[5]:



# In[653]:


#%%capture --no-display
os.chdir('C:\\Users\\moham\\Downloads\\target\\pdbs\\train\\')
from Bio import SeqIO
folder = glob('*')
#k = []
k1 = []
A2 = []
s=0
for i in folder:
    os.chdir('C:\\Users\\moham\\Downloads\\target\\pdbs\\train\\'+str(i)) 
    parser = PDBParser()
    print(i)
 

    pdb_files = glob('*.pdb')
    print(s)
    s=s+1
    S = 0
    for fileName in pdb_files:
        if fileName == str(i)+'_relaxed.pdb' :

            structure_id = fileName[:-12]
            structure = parser.get_structure(structure_id, fileName)
            model = structure[0]
            res_list = Selection.unfold_entities(model, 'R')
            G, B = two_maps(res_list)
            C= two_mapsN(res_list)
            D= two_mapsC(res_list)
            E= two_mapsCB(res_list)
#print("Model %s Chain %s" % (str(model.id), str(chain.id)))
            for chain in model:
                poly = Polypeptide.Polypeptide(chain) 
                a= poly.get_phi_psi_list()
            g= []
            for j in range(len(a)):
                g.append(list(a[j]))
            q=np.array(g)
        #b[1].append(q)
            #a_wild = a_wild[res_mut-5:res_mut+6]            
            k1.append([structure_id.upper(), G,B,C,D,E,q])
    S += 1
   # print(S)
    #np.save('Final.npy',two_matrix)


os.chdir('C:\\Users\\moham\\Downloads\\target\\pdbs\\train')
folder = glob('*')
k1 = []
#d=[]
for i in folder:
    os.chdir('C:\\Users\\moham\\Downloads\\target\\pdbs\\train\\'+str(i)) 

    j=[]
    S = 0
    pdb_files = glob('*.pdb')
    for seq in pdb_files:
        PH=[]
        PS=[]
        OM=[]
        CH=[]
        if seq == str(i)+'_relaxed.pdb':
            structure_id = seq[:-12]
            u = mda.Universe(seq)                       # read atoms and coordinates from PDB or GRO
            protein = u.select_atoms('protein')
            print(structure_id)

            
            omegas = [res.omega_selection() for res in protein.residues]
            phi = [res.phi_selection() for res in protein.residues]
            psi = [res.psi_selection() for res in protein.residues]
            chi1 = [res.chi1_selection() for res in protein.residues]
            if len(omegas)==len(psi)==len(phi)==len(chi1):
                for e in range(len(omegas)):
                    if phi[e] is None:
                        PH.append(0)
                        PH.append(1)
                    else:
                        PH.append(math.cos(phi[e].dihedral.value()))
                        PH.append(math.sin(phi[e].dihedral.value()))
                    if psi[e] is None:
                        PS.append(0)
                        PS.append(1)
                    else:
                        PS.append(math.cos(psi[e].dihedral.value()))
                        PS.append(math.sin(psi[e].dihedral.value()))
                    if chi1[e] is None:
                        CH.append(0)
                        CH.append(1)
                    else:
                        CH.append(math.cos(chi1[e].dihedral.value()))
                        CH.append(math.sin(chi1[e].dihedral.value()))
                    if omegas[e] is None:
                        OM.append(0)
                        OM.append(1)
                    else:
                        OM.append(math.cos(omegas[e].dihedral.value()))
                        OM.append(math.sin(omegas[e].dihedral.value()))
         
        #b[1].append(q)
            #a_wild = a_wild[res_mut-5:res_mut+6]            
            k1.append([structure_id.upper(), PH,PS,OM,CH])
        S += 1



# In[686]


# In[656]:



# In[713]:


w=[]
w1=[]

for i in range(len(e)):
    n0=e[i][0]
    m0=e1[i][0]
    n= e[i][1][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    m= e1[i][1][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    n1=e[i][2][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    m1=e1[i][2][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    n2=e[i][3][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    m2=e1[i][3][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    n3=e[i][4][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    m3=e1[i][4][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    n4=e[i][5][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    m4=e1[i][5][s[i]-5:s[i]+6,s[i]-5:s[i]+6]
    n5=e[i][6][s[i]-5:s[i]+6]
    m5=e1[i][6][s[i]-5:s[i]+6]
    w.append([n0,n,n1,n2,n3,n4,n5])
    w1.append([m0,m,m1,m2,m3,m4,m5])


# In[25]:


e1 = []
for i in range(len(d)):
    for j in range(len(a)):
        if d[i][0]==a[j][0]:
            e1.append(a[j])


# In[36]:


len(c)


# In[700]:


s=[]
m=[]
m1=[]
#l=[]
for i in range(len(r1)):
    for j in range(len(r)):
        if r[j][0][0:5]==r1[i][0]:
            for y in range(len(r1[i][16])):
                if r1[i][16][y]!=r[j][16][y] or r1[i][5][y]!=r[j][5][y] or r1[i][6][y]!=r[j][6][y] or r1[i][8][y]!=r[j][8][y] or r1[i][15][y]!=r[j][15][y]:
                    s.append(y)
                    m.append(r[j])
                    m1.append(r1[i])
               # else:
                #    if r[j][0] not in l:
                 #       l.append(r[j][0])
                    





e1=[]
e2=[]
e3=[]
e4=[]
e5=[]
e6=[]
e7=[]
e8=[]
e9=[]
e10=[]
e11=[]
e12=[]
e13=[]
e14=[]
e15=[]
e16=[]
e17=[]
e18=[]
e19=[]
e20=[]
e21=[]
e22=[]
e23=[]
e24=[]
e25=[]
e26=[]
e27=[]
e28=[]
e29=[]
for i in range(len(a5)):
    e1.append(d5[i][1][0])
    e2.append(d5[i][1][1])
    e3.append(d5[i][1][2])
    e4.append(d5[i][1][3])
    e5.append(d5[i][1][4])
    e6.append(d5[i][1][5])
   # e7.append(d[i][1][8])
    e8.append(a5[i][1][0])
    e9.append(a5[i][1][1])
    e10.append(a5[i][1][2])
    e11.append(a5[i][1][3])
    e12.append(a5[i][1][4])
    e13.append(a5[i][1][5])
    e14.append(a5[i][1][6])
    e15.append(a5[i][1][7])
    e16.append(a5[i][1][8])
    e17.append(a5[i][1][9])
    e18.append(a5[i][1][10])
    e19.append(a5[i][1][11])
    e20.append(a5[i][1][12])
    e21.append(a5[i][1][13])
    e22.append(a5[i][1][14])
    e23.append(a5[i][1][15])
    #e24.append(W_In[i][17])
    e24.append(b5[i][1])
    e25.append(b5[i][2])
    e26.append(b5[i][3])
    e27.append(b5[i][4])
    e28.append(b5[i][5])
    e29.append(b5[i][6])


# In[217]:


W_Node=[]
#q=[]
for i in range(2399,len(d5)):
    print(i)
    #q.append(i)
    
    W_Node.append(np.concatenate((e1[i],e2[i],e3[i],e4[i],e5[i],e6[i],e8[i],e9[i],e10[i],e11[i],e12[i],e13[i],e14[i],e15[i],e16[i],e17[i],e18[i],e19[i],e20[i],e21[i],e22[i],e23[i]),axis=1))






