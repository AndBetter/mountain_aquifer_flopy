# =============================================================================
# modello idrogeologico modflow/flopy che simula in stazionario la falda in una topografia complessa.
# Prende in input ricarica ed eventualmente evapotraspirazione. I flussi in uscita 
# sono simulati con il pacchetto dreno assunto distribuito in corrispondenza della topografia. In questo modo
# si simulano le seepage faces in corrispondenza dei punti di emersione della falda (non noti a priori)
# Bisogna dare in input il DTM della zona più un raster che funge da maschera per il bacino e identifica le celle
# a cui assegnare i dreni (in questo caso tutte). 
# # la griglia di calcolo ha layers orizzontali e i dreni vengono assegnati alla quota della topografia
# ==
# questa versione usa una discretizzazione a parallelepipedo e le celle fuori dal bacino o al 
# di sopra di esso vengono disattivate(regional_groundwater_model_MIO invece ha ladiscretizzazione 
# che segue la topografia)
#
#Il modluo finale fa anche il particle tracking delle particelle distribuite uniformemente sulla supericie 
#attraverso la ricarica.
#Calcola poi il tempo di residenza delle particelle e l'età dello streamflow 


import flopy
import flopy.utils.modpathfile as mpf
import os
import shutil
import sys
import flopy.utils.binaryfile as bf
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import multiprocessing


###########################################################################
#Files e Directories ##############################################
###########################################################################

## Directories EXE workstation
exeMODFLOW = "C:/DEV/Exe/MODFLOW-NWT_64.exe"
exeMODPATH = "C:/DEV/Exe/mpath7.exe"

## Directories EXE laptop
#exeMODFLOW = "C:/PostDoc/Acquiferi_trentini/RegionalGroundwaterModelingwithMODFLOWandFlopy/Exe/MODFLOW-NWT_64.exe"
#exeMODPATH = "C:/PostDoc/Acquiferi_trentini/RegionalGroundwaterModelingwithMODFLOWandFlopy/Exe/mpath7.exe"


#Raster paths workstation
demPath = "E:/Post_TN/Rst/DEM_Maso_100m.tif"           
crPath =  "E:/Post_TN/Rst/CR_Maso_100m.tif"   


#Raster paths laptop
#demPath = "C:/PostDoc/Acquiferi_trentini/RegionalGroundwaterModelingwithMODFLOWandFlopy/Rst/DEM_Maso_100m.tif"           
#crPath =  "C:/PostDoc/Acquiferi_trentini/RegionalGroundwaterModelingwithMODFLOWandFlopy/Rst/CR_Maso_100m.tif"   


#############################################################################
########### Parametri su cui iterare le simulazioni##########################
#############################################################################

#nota: i valori di R corrispondono a ricarica adimensionale R/K pari a [0.001	0.002154435	0.004641589	0.01	0.021544347	0.046415888	0.1	0.215443469	0.464158883	1] 
# nel caso in cui si lasci fissa la conducibilità idraulica (mean_y=-16.12 = mean_k=1e-7m/s)


IMP_DEPTH =[1000]                   #[0, 10, 100, 1000]                    # profondità dello strato impermeabilie rispetto alla cella più depressa del dem
R =    [0.0031536,	0.006794225,	0.014637715,	0.031536,	0.067942252,	0.146377145,	0.31536,	0.679422524,	1.463771455,	3.1536]                     #logscale: [0.0031536,	0.006794225,	0.014637715,	0.031536,	0.067942252,	0.146377145,	0.31536,	0.679422524,	1.463771455,	3.1536]          #logscale: [0.0005, 0.0014, 0.0043, 0.0130, 0.0389, 0.1168, 0.3504, 1.0512, 3.1536]  # R: [0.02, 0.06325, 0.2, 0.6325, 2]             # ricarica
MEAN_Y = [-16.12]                             #[-18.42, -17.27,  -16.12, -14.97 , -13.82, -12.67,  -11.51]                   #[-18.42, -16.12, -13.82, -11.51]      # media del campo log(k)
VAR_Y = [0]                                   # [0, 0, 0, 0, 0, 0, 0]                        #[0, 0, 0, 0]                          # varianza del campo log(k)
TOPOGRAPHY_FACTOR = [1]             #[0.04,0.25, 1, 4]                      # parametro che moltiplica le quote del dem per generare topografie più o meno marcate
ALPHA = [0.0001]                 #[0, 0.0001, 0.001, 0.01]              # parametro che controlla la decrescita esponenziale della Ks
    
porosity=0.1                          
                   
layer_number=100                             # number of layers (se si cambia bisonga cambiare anche nel generatore dei campi random)
particle_nubmer=11283                        # da cambiare se si cambia il numero di particelle e se si cambia il DTM (cosi ce n'è una per cella)




##############################################################################
######### Funzione che fa girare il modello e il particle tracking etc
##############################################################################

def MODELLO(crData, demData,IMP_DEPTH,TOPOGRAPHY_FACTOR,ALPHA,MEAN_Y,VAR_Y,R):     
    
        root_path= os.path.dirname(os.path.abspath(__file__))
        os.chdir(root_path)
        path= os.path.join(root_path,'output_parallelizzato',str(R))    # deve esserci una cartella 'output_parallelizzato' nella stessa directory dove c'e' lo script
        
        if os.path.exists(path):
         shutil.rmtree(path)
         
        os.mkdir(path)
        os.chdir(path)
        
        
        ############################################
        #Initialize Modflow Nwt solver (loop dependent, altrimenti metterlo sopra ai loops)
        ###########################################
        
        modelname = "model_parallelizzato" 
        modelpath = path
        
        
        # Inizializza il solutore
        mf1 = flopy.modflow.Modflow(modelname, exe_name= exeMODFLOW, version="mfnwt", model_ws=modelpath)
        nwt = flopy.modflow.ModflowNwt(mf1 , maxiterout=15000,  maxitinner=10000, mxiterxmd = 10000, headtol=0.001, fluxtol=R/50, linmeth=1, stoptol=1e-10, hclosexmd =1e-3, dbdtheta = 0.5, backflag=1, msdr=25, thickfact=1e-04)

    
        ######################################################################
        # spatial discretization
        ######################################################################
        
        #Boundaries for Dis = Create discretization object, spatial/temporal discretization
        
        nrow = demDs.RasterYSize
        ncol = demDs.RasterXSize
        delr = geot[1]
        delc = abs(geot[5])
        
        
        demData_stretched= TOPOGRAPHY_FACTOR * demData + 1
        crData[crData<0]=0
        demData_stretched[demData_stretched<0]=0
        
        
        ztop = np.ones((nrow , ncol)) *   np.max(demData_stretched[crData>0])       #<------------
        zbot = np.ones((nrow , ncol)) *   np.min(demData_stretched[crData>0]) - IMP_DEPTH
        nlay = layer_number
        delv = (ztop - zbot) / nlay
        botm = np.linspace(ztop, zbot, nlay + 1)
         
          
    
        ################################################################################
        #definition of flow packages 
        ################################################################################
        
        # Array of hydraulic heads per layer
        
        
        #hk = 1e-4
        #vka = 1e-4
        #sy = 0.1
        #ss = 1.0e-4
        
        
        #hk = np.ones((nlay)) * 1e-7
        #hk= np.ones((nlay,nrow,ncol), dtype=np.float32)  *  1e-7
        
        
        # importa dal file field.dat il campo delle hk generato dallo script "Generate_hk_field"
        idec = 2
        if idec != 0:
            hk = np.zeros((nlay,nrow,ncol),dtype=np.float64)
        
            f = open(os.path.join(root_path,'field.dat'), 'rb')
            dt = np.dtype('f8')
            hk = np.fromfile( f, dtype=dt, count=-1 ).reshape((nlay,nrow,ncol),order='C')
            f.close()
        
        hk= np.exp(np.sqrt(VAR_Y)* hk + MEAN_Y)  # passa dalla log conductivity normalizzata alla conducibilità vera e propria  <----------------
         
        #########################################################################
        # fa variare la conducibilità idraulica con la profondità (la matrice reduction_factor_Ks corrisponde a come le k vengono scalate, decresce esponenzialmente da 1 a un valore minimo ora impostato a 0.01)
        #######################################################################
        reduction_factor_Ks = np.ones(hk.shape, dtype=np.float32)
        for idx1 in range(nrow):    
         for idx2 in range(ncol):
          for idx3 in range(nlay):   
           if demData_stretched[idx1,idx2] >= botm[idx3+1,idx1,idx2] and demData_stretched[idx1,idx2]>0:    
            reduction_factor_Ks[idx3,idx1,idx2] = 0.05 + (1-0.05) * np.exp(- ALPHA * (demData_stretched[idx1,idx2] - ( botm[idx3,idx1,idx2] + botm[idx3+1,idx1,idx2] )/2  )  )
        
        hk= np.multiply(hk,reduction_factor_Ks)
        
       ######################################################################################## 
        
        laytyp=np.ones((nlay), dtype=int)
        
        
        # Variables for the DIS package
        dis = flopy.modflow.ModflowDis(mf1, nlay,nrow,ncol,delr=delr,delc=delc,top=ztop,botm=botm[1:],itmuni=1)
        
        # Variables for the BAS package
        iboundData = np.zeros(demData.shape, dtype=np.int32)
        iboundData[crData > 0 ] = 1
        
        
        #condizini iniziali di primo tentativo
        
        #strt= zbot + IMP_DEPTH + (ztop - zbot - IMP_DEPTH) * R/365/86400 / np.mean(hk) 
        #strt= demData_stretched * R/365/86400 / np.mean(hk) + 1
        strt= demData_stretched * 0.5 + 200
        #strt= demData_stretched * 1/(1 + np.exp(5-10*R/365/86400 / np.mean(hk))) + 100
        #strt= zbot + Imp_depth - 10
        #strt= demData_stretched
        
        bas = flopy.modflow.ModflowBas(mf1,ibound=iboundData,strt=strt, hnoflo=-2.0E+020)    # <-------------
        
        
        # Add UPW package to the MODFLOW model
        
        upw = flopy.modflow.ModflowUpw(mf1, laytyp = laytyp, hk = hk, ipakcb=53)
        
        
        #Add the recharge package (RCH) to the MODFLOW model
        rch = np.ones((nrow, ncol), dtype=np.float32) * R/365/86400          #  <--------------------
        rch_data = {0: rch}
        rch = flopy.modflow.ModflowRch(mf1, nrchop=3, rech =rch_data)
        
        # =============================================================================
        # #Add the evapotranspiration package (EVT) to the MODFLOW model
        # evtr = np.ones((nrow, ncol), dtype=np.float32) * 0.1/365/86400
        # evtr_data = {0: evtr}
        # evt = flopy.modflow.ModflowEvt(mf1,nevtop=1,surf=ztop,evtr=evtr_data, exdp=0.5)
        # =============================================================================
        
        
        #Add the drain package (DRN) to the MODFLOW model
        sorgenti = np.zeros(demData.shape, dtype=np.int32)
        sorgenti[crData >0 ] = 1
        lista = []
        for i in range(sorgenti.shape[0]):
            for q in range(sorgenti.shape[1]):
                
                for j in range(nlay):
                 if   demData_stretched[i,q] < botm[j,i,q] and demData_stretched[i,q] > botm[j+1,i,q] and sorgenti[i,q]>0: 
                  w=j
                
                
                if sorgenti[i,q] == 1:
                    #C=hk[w,i,q]*delc*delr/0.1
                    C=1   # diviso per 1e-8 nel caso in cui si usasse la ricarica adimensionale --- ATTENZIONE QUA
                    lista.append([w,i,q,demData_stretched[i,q], C ]) #layer,row,column,elevation(float),conductance  <--------------  
        rivDrn = {0:lista}
        
        drn = flopy.modflow.ModflowDrn(mf1,ipakcb=53, stress_period_data=rivDrn, filenames=None)
        
        
        # Add OC package to the MODFLOW model
        
        #spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
        #oc = flopy.modflow.ModflowOc(mf1, stress_period_data=spd, compact=True)
       
        #oc = flopy.modflow.ModflowOc(mf1)
        
        flopy.modflow.ModflowOc(mf1, stress_period_data={(0, 0): ['save head',
                                                            'save budget',
                                                            'print head']})
        
        
        
        ########################################################################
        ####### writes files and run simulation
        #######################################################################
        
        #Write input files -> write file with extensions
        mf1.write_input()
        
        #run model -> gives the solution
        mf1.run_model()
    
    
        ##########################################
        #legge il file dei potenziali e dei flussi dai dreni
        ###########################################
        
        fname = os.path.join(modelpath, modelname + ".hds")
        hds = bf.HeadFile(fname)
        times = hds.get_times()
        
        fname = os.path.join(modelpath, modelname + ".cbc")
        cbb = bf.CellBudgetFile(fname)
        kstpkper_list = cbb.get_kstpkper()
        
        drain_fluxes_3D= cbb.get_data(kstpkper=(0,0), text='DRAIN',full3D=True)
        drain_fluxes_2D= np.sum(drain_fluxes_3D[0], axis=0)
        drain_fluxes_2D=drain_fluxes_2D/(delc*delr)                         # passa da L^3/T a L/T (da portata a flusso)
        
        drain_fluxes_2D_ARRAY[run_count,:,:]=drain_fluxes_2D
        
        
    
        #######################################################################
        #######################################################################
        #### particle tracking with modpath 7
        #######################################################################
        #######################################################################
        
        
        ###################################################################################
        # create particles distirbuite uniformemente (ricarica)--per il tracking FORWARD##
        ###################################################################################
        
        fname = os.path.join(modelpath, 'model_parallelizzato.hds')
        hdobj = flopy.utils.HeadFile(fname)
        head = hdobj.get_data()
        head_0=head[0,:,:]
        
        plocs = []
        pids = []
        particle_count=0
        for idx1 in range(nrow):    # le particelle vengono assegnate per ogni cella alla quota corrispondente alla piezomatrica 
         for idx2 in range(ncol):   
           if demData_stretched[idx1,idx2] >0: 
              plocs.append((0, idx1, idx2))
              pids.append(particle_count)
              particle_count=particle_count+1
            
        part0 = flopy.modpath.ParticleData(plocs, drape=1, structured=True, particleids=pids)
        pg0 = flopy.modpath.ParticleGroup(particlegroupname='PG1', particledata=part0,filename='ex01a.pg1.sloc')
        
                
        particlegroups = [pg0]
        
        
        # default iface for MODFLOW-2005 and MODFLOW 6
        defaultiface = {'RECHARGE': 6, 'ET': 6}
        defaultiface6 = {'RCH': 6, 'EVT': 6}
        
        
        # create modpath files
        
        mp = flopy.modpath.Modpath7(modelname=modelname + '_mp', flowmodel=mf1, exe_name = exeMODPATH, model_ws=modelpath)
        
        mpbas = flopy.modpath.Modpath7Bas(mp, porosity=porosity,defaultiface=defaultiface)
        
        mpsim = flopy.modpath.Modpath7Sim(mp, simulationtype='combined',
                                          trackingdirection='forward',     
                                          weaksinkoption='stop_at',
                                          weaksourceoption='stop_at',
                                          budgetoutputoption='summary',
                                          budgetcellnumbers=None,
                                          traceparticledata=None,
                                          referencetime=[0, 0, 0.],
                                          stoptimeoption='extend',
                                          timepointdata=[500, 1000.],
                                          zonedataoption='off', zones=None,
                                          particlegroups=particlegroups)
        
        # write modpath datasets
        mp.write_input()
        
        # run modpath
        mp.run_model()
        
        
        ###########################
        # get pathline file (file molto voluminoso, commentare se non necessario)
        ###############################
        #import flopy.utils.modpathfile as mpf                             
        pthobj = flopy.utils.PathlineFile(modelname + '_mp'+'.mppth')
        p = pthobj.get_alldata()          # pathfile per tutte le particelle
        #p1 = pthobj.get_data(partid=1)   # pathfile per una particlella specificata
        
        
        # Calcola la lunghezza dei flowpaths a partire dall'oggetto pathfile
        flowpath_lengths=np.zeros( (len(p),), dtype=np.float32)

        for i in range(len(p)):                                                  # itera su tutti i flowpaths
            coord_array=np.array([p[i].x,p[i].y,p[i].z]).T                       # crea array nx3 con i vertici della traiettoria di un flowpat
            length = np.sqrt(np.sum(np.diff(coord_array, axis=0)**2, axis=1))    # calcola lunghezza singolo flowpath
            flowpath_lengths[i] = np.sum(length)
        
        
        ###########################
        # get travel times
        ###############################
        
        endobj = flopy.utils.EndpointFile(modelname + '_mp'+'.mpend')   # ottiene tutti i travel times
        e = endobj.get_alldata()
        traveltime=e.time
        #e1 = endobj.get_data(partid=1)   # travel time per una particlella specificata
        
        
        traveltime_ARRAY[:,run_count]=traveltime
        #traveltime_ARRAY.append=traveltime
        
        ##########################################################
        ###### estrae le traveltimes in corrispondenza dei punti in cui il flusso esce dal pc e calcola l'età dello streamflow
        #########################################################
        
        endobj = flopy.utils.EndpointFile(modelname + '_mp'+'.mpend')  
        temp=np.where(drain_fluxes_3D[0].data<0)
    
    
        flux=[]
        time_particles_at_cell=[]
        streamflow_age=[]
        
        for i in range(len(temp[0])): 
            flux.append(drain_fluxes_3D[0].data[temp[0][i], temp[1][i], temp[2][i]]) 
            e0 = endobj.get_destination_endpoint_data(dis.get_node((temp[0][i], temp[1][i], temp[2][i])))  # dis.get_node ottiene a partire dalla terna k,i,j il codice identificativo univoco della cella 
            time_particles_at_cell.append(e0.time) 
    
    
        for i in range(len(flux)):  # cambia di segno ai flussi (flussi sono in m3/s)
            flux[i]=flux[i] * -1
    
        for j in range(len(time_particles_at_cell)):    #calcola le età pesate con i flussi
         for jj in range(len(time_particles_at_cell[j])):
            streamflow_age.append(time_particles_at_cell[j][jj]*flux[j] / sum(flux) )
            
        streamflow_age=np.array(streamflow_age)     
            
            
        streamflow_age_ARRAY[0:len(streamflow_age),run_count]=streamflow_age
    
        R_K_ratio[0,run_count] = ( R/ 3600/24/365) /  np.exp(  (MEAN_Y + VAR_Y)/2)
    
    
        #########################################################################################
        ####### end particle tracking ###########################################################
        #########################################################################################

        return traveltime, streamflow_age, flowpath_lengths, drain_fluxes_2D




###########################################################################
#create basic modflow object ##############################################
###########################################################################



##########################################################################
#open and read raster files ##############################################
##########################################################################


#Open files
demDs =gdal.Open(demPath)
crDs = gdal.Open(crPath)
geot = crDs.GetGeoTransform() #Xmin, deltax, ?, ymax, ?, delta y

geot


# Get data as arrays
demData_original = demDs.GetRasterBand(1).ReadAsArray()
crData = crDs.GetRasterBand(1).ReadAsArray()

demData=np.array(demData_original, copy=True)  
demData[crData>0]=demData[crData>0]-np.min(demData[crData>0])  # shifta il dem in modo che l'outlet abbia quota 0 (in realtà 1 per motivi dei calcoli successivi)


# Get statistics
stats = demDs.GetRasterBand(1).GetStatistics(0,1) 
stats


##########################################################################

run_count=0
risultati=[]

drain_fluxes_2D_ARRAY=np.zeros( (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA),) + demData.shape, dtype=np.float32)  # crea array vuoti che verranno riempiti durante i loop sui parametri 
traveltime_ARRAY=np.zeros( ((particle_nubmer),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
streamflow_age_ARRAY=np.zeros( ((particle_nubmer),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
flowpath_lengths_ARRAY=np.zeros( ((particle_nubmer),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
R_K_ratio=np.zeros( ((1),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)

for iter_1 in range(len(IMP_DEPTH)):
    
 for iter_2 in range(len(TOPOGRAPHY_FACTOR)): 

  for iter_3 in range(len(ALPHA)):   

   for iter_4 in range(len(MEAN_Y)): 
  

    def main():
      pool = multiprocessing.Pool(len(R))
      risultati = pool.starmap(MODELLO, [(crData, demData,IMP_DEPTH[iter_1],TOPOGRAPHY_FACTOR[iter_2],ALPHA[iter_3],MEAN_Y[iter_4],VAR_Y[iter_4],ricarica) for ricarica in R])

      pool.close()
      pool.join()

      return risultati

        

    if __name__ == '__main__':
     # Better protect your main function when you use multiprocessing
    
    
     risultati = risultati + main()
    
     
#costuisce gli array degli outputs a partire dalle liste in "risutlati"

for i in range(len(risultati)):
 traveltime_ARRAY[0:len(risultati[i][0]),i]=risultati[i][0]
 streamflow_age_ARRAY[0:len(risultati[i][1]),i]=risultati[i][1]
 flowpath_lengths_ARRAY[0:len(risultati[i][2]),i]=risultati[i][2]
 drain_fluxes_2D_ARRAY[i,:,:]=risultati[i][3]

     

        
     # print(run_count)
     # run_count=run_count+1
     
        
##################################################################################
#######end model runs#############################################################
##################################################################################





# =============================================================================
# #######################################################################
# ## model results and post processing###################################
# #######################################################################
# 
# 
# #Import model
# ml = flopy.modflow.Modflow.load('../Model_mio/model_parallelizzato.nam')    #<---------
# 
# # First step is to set up the plot
# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# 
# # Next we create an instance of the ModelMap class
# modelmap = flopy.plot.ModelMap(sr=ml.dis.sr)
# 
# # Then we can use the plot_grid() method to draw the grid
# # The return value for this function is a matplotlib LineCollection object,
# # which could be manipulated (or used) later if necessary.
# linecollection = modelmap.plot_grid(linewidth=0.4)
# 
# 
# 
# #Cross section of model grid representation
# 
# fig = plt.figure(figsize=(15, 6))
# ax = fig.add_subplot(1, 1, 1)
# # Next we create an instance of the ModelCrossSection class
# #modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Column': 5})
# modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Row': 99})
# 
# # Then we can use the plot_grid() method to draw the grid
# # The return value for this function is a matplotlib LineCollection object,
# # which could be manipulated (or used) later if necessary.
# linecollection = modelxsect.plot_grid(linewidth=0.4)
# t = ax.set_title('Column 6 Cross-Section - Model Grid')
# 
# 
# 
# #Active/inactive cells on model extension
# 
# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# modelmap = flopy.plot.ModelMap(model=ml, rotation=0)
# quadmesh = modelmap.plot_ibound(color_noflow='cyan')
# linecollection = modelmap.plot_grid(linewidth=0.4)
# 
# 
# #Cross sections of active/inactive cells
# 
# fig = plt.figure(figsize=(15, 6))
# ax = fig.add_subplot(1, 1, 1)
# modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Column': 5})
# patches = modelxsect.plot_ibound(color_noflow='cyan')
# linecollection = modelxsect.plot_grid(linewidth=0.4)
# t = ax.set_title('Column 6 Cross-Section with IBOUND Boundary Conditions')
# 
# 
# #Channel network as drain (DRN) package
# 
# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# modelmap = flopy.plot.ModelMap(model=ml, rotation=-14)
# quadmesh = modelmap.plot_ibound(color_noflow='cyan')
# quadmesh = modelmap.plot_bc('DRN', color='blue')
# linecollection = modelmap.plot_grid(linewidth=0.4)
# 
# 
# #Model grid and heads representation
# 
# fname = os.path.join(modelpath, 'model_parallelizzato.hds')
# hdobj = flopy.utils.HeadFile(fname)
# head = hdobj.get_data()
# 
# fig = plt.figure(figsize=(30, 30))
# ax = fig.add_subplot(1, 2, 1, aspect='equal')
# modelmap = flopy.plot.ModelMap(model=ml, rotation=-14)
# #quadmesh = modelmap.plot_ibound()
# quadmesh = modelmap.plot_array(head, masked_values=[-2.e+20], alpha=0.8)
# linecollection = modelmap.plot_grid(linewidth=0.2)
# 
# #############################################################
# #Plot fluxes out of drains 3D (fette)#######################
# ############################################################
# 
# a=drain_fluxes_3D[0]
# a=a.data[40,:,:]
# fig=plt.imshow(a, interpolation='none',vmin=-0.005, vmax=0)
# plt.show()
# 
# #############################################################################
# #Plot fluxes out of drains 2D (somma lungo le verticali)#####################
# #############################################################################
# 
# ##plot singolo
# 
# fig=plt.imshow(drain_fluxes_2D, interpolation='none')  # ...interpolation='none',vmin=-0.01, vmax=0
# plt.show()
# 
# 
# 
# ##plotta tutti
# 
# # =============================================================================
# # plt.figure()
# # f, axarr = plt.subplots(2,5) 
# # 
# # axarr[0,0].imshow(drain_fluxes_2D_ARRAY[0,:,:], interpolation='none')  
# # axarr[0,1].imshow(drain_fluxes_2D_ARRAY[1,:,:], interpolation='none')  
# # axarr[0,2].imshow(drain_fluxes_2D_ARRAY[2,:,:], interpolation='none')  
# # axarr[0,3].imshow(drain_fluxes_2D_ARRAY[3,:,:], interpolation='none')  
# # axarr[0,4].imshow(drain_fluxes_2D_ARRAY[4,:,:], interpolation='none')  
# # axarr[1,0].imshow(drain_fluxes_2D_ARRAY[5,:,:], interpolation='none')  
# # axarr[1,1].imshow(drain_fluxes_2D_ARRAY[6,:,:], interpolation='none')  
# # axarr[1,2].imshow(drain_fluxes_2D_ARRAY[7,:,:], interpolation='none')  
# # axarr[1,3].imshow(drain_fluxes_2D_ARRAY[8,:,:], interpolation='none')  
# # axarr[1,4].imshow(drain_fluxes_2D_ARRAY[9,:,:], interpolation='none')
# # 
# # 
# # =============================================================================
# 
# 
# ############################################
# ##### saves variables
# ############################################
# 
# import shelve
# 
# filename='C:/PostDoc/Acquiferi_trentini/Risultati/AAA_adimensionali_full_run/shelve.out'
# my_shelf = shelve.open(filename,'n') # 'n' for new
# 
# my_shelf['demData_original'] = globals()['streamflow_age_ARRAY']
# my_shelf['drain_fluxes_2D_ARRAY'] = globals()['drain_fluxes_2D_ARRAY']
# my_shelf['traveltime_ARRAY'] = globals()['traveltime_ARRAY']
# my_shelf['streamflow_age_ARRAY'] = globals()['streamflow_age_ARRAY']
# 
# 
# my_shelf.close()
# 
# 
# ############################################
# ##### load variables
# ############################################
# 
# # =============================================================================
# # import shelve
# # 
# # filename='C:/PostDoc/Acquiferi_trentini/Risultati/AAA_adimensionali_full_run/shelve.out'
# # 
# # my_shelf = shelve.open(filename)
# # for key in my_shelf:
# #     globals()[key]=my_shelf[key]
# # my_shelf.close()
# # =============================================================================
# =============================================================================
