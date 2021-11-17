"""
IMINFECTOR algorithm
"""
import pandas as pd
import numpy as np
import os
import json
import time

def softmax(x):
        return np.exp(x)/np.sum(np.exp(x))

class IMINFECTOR:
    def __init__(self, fn, embedding_size):
        self.fn = fn
        self.embedding_size = embedding_size
        self.file_Sn = fn.capitalize()+"/Embeddings/infector_source3.txt" 
        self.file_Tn = fn.capitalize()+"/Embeddings/infector_target3.txt"
        if(fn=="digg"):
            self.size=50
            self.P = 40
        elif(fn=="weibo"):
            self.size=1000
            self.P = 10
        else:
            self.size=10000
            self.P = 10
            
    def infl_set(self,candidate,size,uninfected):
        return np.argpartition(self.D[candidate,uninfected],-size)[-size:]
    
    def infl_spread(self,candidate,size,uninfected):
        return sum(np.partition(self.D[candidate,uninfected], -size)[-size:])
        
    def embedding_matrix(self,var):
        """
        Derive the matrix embeddings vector from the file
        """
        if var=="T":
            embedding_file= self.file_Tn
            embed_dim = [self.target_size,self.embedding_size]
        else:
            embedding_file= self.file_Sn
            embed_dim = [self.input_size,self.embedding_size]

        nodes = []
        f = open(embedding_file,"r")
        emb = np.zeros((embed_dim[0],embed_dim[1]), dtype=np.float)
        i=0
        for l in f:
            if "[" in l:
                combined = ""
            if "]" in l:
                combined = combined+" "+l.replace("\n","").replace("[","").replace("]","")
                parts = combined.split(":")
                nodes.append(int(parts[0]))
                emb[i] = np.asarray([float(p.strip()) for p in parts[1].split(" ") if p!=""],dtype=np.float)
                i+=1
            combined = combined+" "+l.replace("\n","").replace("[","").replace("]","")
        return nodes, emb
        
    def read_sizes(self):
        f = open(self.fn.capitalize()+"/"+self.fn+"_sizes.txt","r")
        self.target_size = int(next(f).strip())
        self.input_size = int(next(f).strip())
        f.close()
        
    def get_influencer_fairness(self, init_idx):
        influencer_fairness_read = open(self.fn.capitalize()+"/Init_Data/influencer_fairness.txt","r")
        gender_dict_read = {}
        for line in influencer_fairness_read:
            data = line.replace("\n","").split(",")
            gender_dict_read[data[0]] = data[1]
        influencer_fairness_read.close()    
        
        influencer_fairness_list = []
        for n in init_idx:
            if str(n) in gender_dict_read:
                influencer_fairness_list.append(float(gender_dict_read[str(n)]))
            else:
                raise Exception("Sorry, {} is missing from train_cascade".format(n))
         
        return np.array(influencer_fairness_list)                
            

    def compute_D(self,S,T,nodes_idx,init_idx, influencer_fairness_array):
        """
        # Derive matrix D and vector E
        """
        print(S.shape[0])
        perc = int(self.P*S.shape[0]/100)       
        norm = np.apply_along_axis(lambda x: sum(x**2),1,S)
        norm = (norm - min(norm))/( max(norm) - min(norm))
         ############# CHANDU: EXPERIMENT ###########
        norm = np.multiply(norm, influencer_fairness_array)        
        ############# CHANDU: EXPERIMENT ###########      
        self.chosen = np.argsort(-norm)[0:perc]
        norm = norm[self.chosen]
        bins = (self.target_size)*norm/sum(norm)
        self.bins = [int(i) for i in np.rint(bins)]
        np.save(self.fn.capitalize()+"/E", self.bins)
        S = S[self.chosen] 
        
        self.D = np.dot(np.around(S,4),np.around(T.T,4))  
    
    def process_D(self):
        """
        # Derive the diffusion probabilities. Had to be separated with compute_D, beause of memory
        """
        self.D = np.apply_along_axis(lambda x:x-abs(max(x)), 1, self.D) 
        self.D = np.apply_along_axis(softmax, 1, self.D) 
        self.D = np.around(self.D,3)
        self.D = abs(self.D)
        np.save(self.fn.capitalize()+"/D",self.D)
     
    def run_method(self,init_idx):
        """
        # IMINFECTOR algorithm
        """
        Q = []
        self.S = []   
        nid = 0
        mg = 1
        iteration = 2
        infed = np.zeros(self.D.shape[1])
        total = set([i for i in range(self.D.shape[1])])
        uninfected = list(total-set(np.where(infed)[0]))
        
        #----- Initialization
        for u in range(self.D.shape[0]):
            temp_l = []
            temp_l.append(u)
            spr = self.infl_spread(u,int(self.bins[u]),uninfected)    
            temp_l.append(spr)
            temp_l.append(0)
            Q.append(temp_l)
            
        # Do not sort
        ftp = open(self.fn.capitalize()+"/Seeds/final_seeds.txt","w")  
        idx = 0
        while len(self.S) < self.size :
            u = Q[0]
            new_s = u[nid]
            if (u[iteration] == len(self.S)):
                influenced = self.infl_set(new_s,int(self.bins[new_s]),uninfected)   
                #influenced = self.infl_set(new_s,self.bins[new_s],uninfected)   
                infed[influenced]  = 1         
                uninfected = list(total-set(np.where(infed)[0]))
                
                #----- Store the new seed
                ftp.write(str(init_idx[self.chosen[new_s]])+"\n")
                self.S.append(new_s)
                if(len(self.S)%50==0):
                    print(len(self.S))
                #----- Delete uid
                Q = [l for l in Q if l[0] != new_s]
                
            else:
                #------- Keep only the number of nodes influenceed to rank the candidate seed        
                spr = self.infl_spread(new_s,int(self.bins[new_s]),uninfected)        
                u[mg] = spr
                if(u[mg]<0):
                    print("Something is wrong")
                u[iteration] = len(self.S)
                Q = sorted(Q, key=lambda x:x[1],reverse=True)
            
        ftp.close()
            
        
def run(fn,embedding_size,log):
    f = open(fn.capitalize()+"/Init_Data/train_set.txt","r")
    start = time.time()
    iminfector = IMINFECTOR(fn,embedding_size)

    iminfector.read_sizes()
    
    nodes_idx, T = iminfector.embedding_matrix("T")
    init_idx, S = iminfector.embedding_matrix("S")
    influencer_fairness_array = iminfector.get_influencer_fairness(init_idx)
    
    iminfector.compute_D(S,T,nodes_idx,init_idx, influencer_fairness_array) 
    del T,S,nodes_idx
    iminfector.process_D()
    iminfector.run_method(init_idx)
    
        
