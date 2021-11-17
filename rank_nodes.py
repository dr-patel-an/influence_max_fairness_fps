"""
Take the top nodes ranked based on kcore and avg cascade length (top no=seed size for each dataset)
"""

import os            
import pandas as pd


def run(fn):
    dat = pd.read_csv(fn.capitalize()+"/node_features.csv")
    if(fn =="digg"):
        perc = 100
    elif(fn=="weibo"):
        perc = 100 
    else:
        perc = 10000
    	
    top = pd.DataFrame(columns=dat.columns)
    for col in dat.columns:
        if(col=="Node"):
            continue
        top[col] = dat.nlargest(perc,col)["Node"].values

    top = top.drop(["Node"], axis=1)  
    
    for c in top.columns:
        f = open(fn.capitalize()+"/Seeds/"+c.lower()+"_seeds.txt","w")
        f.write(" ".join([str(x) for x in list(top.loc[0:perc,c].values)]))
        f.close()
    
