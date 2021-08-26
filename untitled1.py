import numpy as np
import multiprocessing 
from multiprocessing import Pool

def somma(a,b):
    x=a+b
    y=a*b
    return(x,y)




# =============================================================================
# for iter in range(12):
# 
#  print(somma(iter,iter,iter+1))
# =============================================================================


R=[5,4,3,2,1]

if __name__ == '__main__':
   for i in R:
    with Pool(5) as p:
       print(p.starmap(somma, [[1, 2], [3,i]]))
       
       
       
