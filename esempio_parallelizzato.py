import multiprocessing
from numpy import random
import numpy as np

def somma(minimo,massimo,N):
    x= sum(random.randint(minimo,massimo,N))/N
    y= sum(random.randint(minimo,massimo,N)**2)/N
    return(x,y)


def main():
    pool = multiprocessing.Pool(16)
    results = pool.starmap(somma, [(-1, mxx, 10000) for mxx in range(16)])

    pool.close()
    pool.join()

    RESULTS=[]
    for result in results:
        # prints the result string in the main process
        print(result)
        RESULTS.append(result)
        
    return RESULTS

        

if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    risultati=main()
    risultati=np.array(risultati)
