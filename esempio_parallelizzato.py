import multiprocessing
from numpy import random
import numpy as np

def somma(minimo,massimo,N):
    x_1= sum(random.randint(minimo,massimo,N))/N
    x_2= sum(random.randint(minimo,massimo,N))/N
    y_1= sum(random.randint(minimo,massimo,N)**2)/N
    y_2= sum(random.randint(minimo,massimo,N)**2)/N
    return(x_1,x_2,y_1,y_2)


def main():
    pool = multiprocessing.Pool(8)
    results = pool.starmap(somma, [(-1, mxx, 100000) for mxx in range(16)])

    pool.close()
    pool.join()
          
    return results
 
         
 
if __name__ == '__main__':
     # Better protect your main function when you use multiprocessing
     risultati=np.stack(main(),axis=1)
     
     X=risultati[0:2,:]
     Y=risultati[2:4,:]




