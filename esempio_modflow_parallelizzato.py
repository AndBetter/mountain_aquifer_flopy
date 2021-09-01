  
def MODELLO(Lx,Ly,ztop,zbot,nlay,nrow,ncol,head_up,head_down):
    import numpy as np
    import flopy
    import matplotlib.pyplot as plt
    import flopy.utils.binaryfile as bf
    import os
     
    
    root_path= os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_path)
    path= os.path.join(root_path,str(head_down))
    os.mkdir(path)
    os.chdir(path)

    

        
    modelname = "prova_modflow_parallelizzato"
    executable= 'C:/Modflow/MF2005.1_12/bin/mf2005.exe'
    mf = flopy.modflow.Modflow(modelname, exe_name=executable)
    
    Lx = Lx
    Ly = Ly
    ztop = ztop
    zbot = zbot
    nlay = nlay
    nrow = nrow
    ncol = ncol
    delr = Lx / ncol
    delc = Ly / nrow
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)
    
    dis = flopy.modflow.ModflowDis(
        mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm[1:])
    
    
    
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] = head_up
    strt[:, :, -1] = head_down
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    
    lpf = flopy.modflow.ModflowLpf(mf, hk=10.0, vka=10.0, ipakcb=53)
    
    spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)
    
    pcg = flopy.modflow.ModflowPcg(mf)
    
    mf.write_input()
    
    success, buff = mf.run_model()
    if not success:
        raise Exception("MODFLOW did not terminate normally.")
        
    hds = bf.HeadFile(f"{modelname}.hds")
    head = hds.get_data(totim=1.0)
    
    # Extract the heads
    hds = bf.HeadFile(f"{modelname}.hds")
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    head_top=head[0,:,:]
    
    return head_top


    
#head, size = MODELLO(100,1000,0,-1000,100,100,1000,0,-100)


import multiprocessing
import numpy as np


Lx=100
Ly=1000
ztop=0
zbot=-1000
nz=100
nx=100
ny=1000
h_up=0
h_down= [-1,-100,-200,-300, -400, -500, -600, -700]


def main(h_down):
    
    pool = multiprocessing.Pool(8)
    head = pool.starmap(MODELLO, [(Lx,Ly,ztop,zbot,nz,nx,ny,h_up,h_) for h_ in h_down])
     
    pool.close()
    pool.join()
          
    return head # HEAD_ARRAY
 
         
 
if __name__ == '__main__':
     # Better protect your main function when you use multiprocessing
     head_array=main(h_down)
     

     
