# -*- coding: utf-8 -*-
"""
INFECTOR neural network
"""
# -*- coding: utf-8 -*-
import os
import time
import math  
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import json
  

class INFECTOR:
    def __init__(self, fn , learning_rate,n_epochs,embedding_size,num_samples):
        self.fn=fn
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.embedding_size = embedding_size
        self.num_samples = num_samples
        self.file_Sn = fn.capitalize()+"/Embeddings/infector_source3.txt" 
        self.file_Tn = fn.capitalize()+"/Embeddings/infector_target3.txt" 
        
    def create_dicts(self):
        """
        # Min max normalization of cascade length and source-target dictionaries
        """
#        f = open(self.fn+"/train_set.txt","r") 
        f = open(self.fn.capitalize()+"/Init_Data/train_set.txt","r")
        initiators = []
        self.mi = np.inf
        self.ma = 0
        for l in f:
            parts  = l.split(",")
            initiators.append(parts[0])
            t = int(parts[2])
            if(t<self.mi):
                self.mi = t
            if(t>self.ma):
                self.ma = t
        self.rang= self.ma-self.mi
        
        #----------------- Source node dictionary
        initiators = np.unique((initiators))
        
        self.dic_in = {initiators[i]:i for i in range(0,len(initiators))}
        f.close()     
        self.vocabulary_size = len(self.dic_in)
        print(self.vocabulary_size)
        #----------------- Target node dictionary
        f = open(self.fn.capitalize()+"/Init_Data/"+self.fn+"_incr_dic.json","r")
        self.dic_out = json.load(f)
        self.target_size = len(self.dic_out)
        print(self.target_size) 	
        f = open(self.fn.capitalize()+"/"+self.fn+"_sizes.txt","w")
        f.write(str(self.target_size)+"\n")
        f.write(str(self.vocabulary_size))
        f.close()

        
    def model(self):
        """
        # The multi-task learning NN to classify influenced nodes and predict cascade length
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            #---- Batch size depends on the cascade
            u = tf.placeholder(tf.int32, shape=[None,1],name="u")
            v = tf.placeholder(tf.int32, shape=[None,1],name="v") 
            
            #------------ Source (hidden layer embeddings)
            S = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="S")
            u2 =  tf.squeeze(u)
            Su = tf.nn.embedding_lookup(S,u2, name="Su")
            #------------- First task
            #------------ Target (hidden and output weights)
            T = tf.Variable(tf.truncated_normal( [self.target_size, self.embedding_size], stddev = 1.0 / math.sqrt(self.embedding_size)), name="T")
            
            #---- Noise contrastive loss function
            nce_biases = tf.Variable(tf.zeros([self.target_size])) 
        
            self.loss1 = tf.reduce_mean(
                tf.nn.nce_loss(weights= T,
                     biases=nce_biases,
                     labels=v,
                     inputs=Su,
                     num_sampled=self.num_samples,
                     num_classes = self.target_size))
            
            #------------- Second task
            #---- Cascade length
            c = tf.placeholder(tf.float32,name="c") 
            
            #------------ Cascade length weights (output layer of cascade length prediction)
            C = tf.constant(np.repeat(1,self.embedding_size).reshape((self.embedding_size,1)),tf.float32, name="C")
            
            #------------ Bias for cascade length
            b_c = tf.Variable(tf.zeros((1,1)),name="b_c")
            
              #------------- Third task # APPROACH 2
            #---- Fainess Score
            fair = tf.placeholder(tf.float32,name="fair") 
            
            #------------ fairness weights (output layer of fainess prediction)
            Fair = tf.constant(np.repeat(1,self.embedding_size).reshape((self.embedding_size,1)),tf.float32, name="Fair")
            
            #------------ Bias for fainrss score
            b_fair = tf.Variable(tf.zeros((1,1)),name="b_fair")
            
            #------------ Loss2  
            alpha = 0.0 # APPROACH 2
            tmp= tf.tensordot(Su,C,1)
            o2 = tf.sigmoid(tmp+b_c)
#             self.loss2 = tf.square(o2-c)
            self.loss2 = alpha * tf.square((o2-c)) #APPROACH2 
            
            
#                 #------------ Loss3   # APPROACH 2
            beta = 1.0 #APPROACH 2
            tmp2 = tf.tensordot(Su,Fair,1)
            o_f_2 = tf.sigmoid(tmp2 + b_fair)
#             self.loss3 = tf.square(o_f_2 - fairness_score)
            self.loss3 = beta * tf.square(o_f_2 - fair) #APPROACH2 
        
            #---- To retreive the embedding-node pairs after training
            n_in = tf.placeholder(tf.int32,shape=[1],name="n_in")
            self.Sn = tf.nn.embedding_lookup(S,n_in,name="Sn")
        
            n_out = tf.placeholder(tf.int32,shape=[1],name="n_out")
            self.Tn = tf.nn.embedding_lookup(T,n_out,name="Tn")
            
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)   
            #--- Seperate optimizations, for joint to loss1+loss2
            self.train_step1 = optimizer.minimize(self.loss1)    
            self.train_step2 = optimizer.minimize(self.loss2)
            self.train_step3 = optimizer.minimize(self.loss3)
        
    def train(self):
        """
	    # Train the model
	    """
        l1s = []
        l2s = []
        l3s = [] # APPROACH 2
	    #sess = tf.InteractiveSession(graph = infector.graph)
        with  tf.Session(graph = self.graph) as sess:
            sess.run(tf.initialize_all_variables()) 
            for epoch in range(self.n_epochs):
                 #--------- Train 
                 f = open(self.fn.capitalize()+"/Init_Data/train_set.txt","r") 
                 idx = 0
                 init= -1
                 inputs = []
                 labels = []
                 #---- Build the input batch
                 for line in f:
                      #---- input node, output node, copying_time, cascade_length, 10 negative samples 
                      sample = line.replace("\r","").replace("\n","").split(",")              
                      try:
                           original = self.dic_in[sample[0]]
                          # print(original)
                           label = self.dic_out[sample[1]]
                      except:
                          continue
					#---- check if we are at the same cascade
                      if(init==original or init<0):
                          init = original
                          inputs.append(original)
                          labels.append(label)
                          casc_len = int(sample[2])
                          casc = (float(sample[2])-self.mi)/self.rang
                          fairness_score = float(float(sample[3]) * 1) 
                          
					#---- New cascade, train on the previous one
                      else:
						#---------- Run one training batch
						#--- Train for target nodes
                           if len(inputs)<2:
                               inputs.append(inputs[0])
                               labels.append(labels[0])
                           inputs = np.asarray(inputs).reshape((len(inputs),1))
                          # print('inputs: ', inputs)
                           labels = np.asarray(labels).reshape((len(labels),1))
                           #print("labels: ", labels)
					
						#------------------ HERE
                           sess.run(self.train_step1, feed_dict = {"u:0": inputs, "v:0": labels, "c:0": [[0]], "fair:0": [[0]]}) 
						
						#--- Train for cascade length
                           sess.run(self.train_step2, feed_dict = {"u:0": inputs[0].reshape(1,1), "v:0": labels, "c:0": [[casc]], "fair:0": [[0]]}) 
                            
                            						#--- Train for fairness score
#                            sess.run(self.train_step3, feed_dict = {"u:0": inputs[0].reshape(1,1), "v:0": labels, "c:0": [[0]], "fair:0": [[fairness_score]]})
                           for i in range(casc_len):
                                  sess.run(self.train_step3, feed_dict = {"u:0": inputs[0].reshape(1,1), "v:0": labels, "c:0": [[0]], "fair:0": [[fairness_score]]})
                        
                        
                           idx+=1
						
                           if idx%1000 == 0: # Collecting losses to see if the values are decreasing  # APPROACH 2
							#loss1.eval(feed_dict = {u: inputs, v: labels, c: [[0]]}) 
                               l1 = sess.run(self.loss1, feed_dict = {"u:0": inputs, "v:0": labels, "c:0": [[casc]], "fair:0": [[fairness_score]]}) 
							#l2 = loss2.eval(feed_dict = {u: inputs[0].reshape(1,1), v: labels, c: [[casc]]}) 
                               l2 = sess.run(self.loss2, feed_dict = {"u:0": inputs[0].reshape(1,1), "v:0": labels, "c:0": [[casc]],"fair:0": [[fairness_score]]}) 
							
                               l3 = sess.run(self.loss3, feed_dict = {"u:0": inputs[0].reshape(1,1), "v:0": labels, "c:0": [[casc]], "fair:0": [[fairness_score]] }) # APPROACH 2
                               l1s.append(l1)
                               l2s.append(l2)
                               l3s.append(l3) # APPROACH 2
                               print('Loss 3 at step %s: %s' % (idx, l3)) # APPROACH 2
                               print('Loss 2 at step %s: %s' % (idx, l2)) 
                               print('Loss 1 at step %s: %s' % (idx, l1))
						   
						#---- Arrange for the next batch
                           inputs = []
                           labels = []
                           inputs.append(original)
                           labels.append(label)
                           casc = (float(sample[2])-self.mi)/self.rang
                           init = original
                 f.close()

            fsn = open(self.file_Sn,"w")
            ftn = open(self.file_Tn,"w")
			
            #---------- Store the source embedding of each node
            for node in self.dic_in.keys():
                 emb_Sn = sess.run("Sn:0",feed_dict = {"n_in:0":np.asarray([self.dic_in[node]])})
                 fsn.write(node+":"+",".join([str(s) for s in list(emb_Sn)])+"\n")
            fsn.close()	
            #---------- Store the target embedding of each node
            for node in self.dic_out.keys():
                emb_Tn = sess.run("Tn:0",feed_dict = {"n_out:0":np.asarray([self.dic_out[node]])})
                ftn.write(node+":"+",".join([str(s) for s in list(emb_Tn)])+"\n")
            ftn.close()
			
            return l1s,l2s,l3s # APPROACH 2
#             return l1s,l2s 

                
def run(fn,learning_rate,n_epochs,embedding_size,num_neg_samples,log):
    start = time.time()
    infector = INFECTOR(fn,learning_rate,n_epochs,embedding_size,num_neg_samples)
    
    infector.create_dicts()
    
    infector.model()
    
#     l1s,l2s = infector.train()
    l1s,l2s,l3s = infector.train()  # APPROACH 2

    log.write("Time taken for the "+fn+" infector:"+str(time.time()-start)+"\n")

