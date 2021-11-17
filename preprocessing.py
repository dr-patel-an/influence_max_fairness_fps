import os
from weibo_preprocessing import weibo_preprocessing
#from digg_preprocessing import digg_preprocessing


"""
Main
"""
if __name__ == '__main__':
    ## Create folder structure
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(os.path.join(dname,"..","Data"))
    
    
		
    ans = weibo_preprocessing(os.path.join("Weibo","Init_Data"))
    #ans = digg_preprocessing(os.path.join("..","Digg","Init_Data"))
    #ans = digg_preprocessing(os.path.join("Digg","Init_Data"))
   # print(ans)
