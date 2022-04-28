# modello idrogeologico modflow/flopy che simula in stazionario la falda in una topografia complessa.
# Prende in input ricarica ed eventualmente evapotraspirazione (spazialmente omogenea al momento). I flussi in uscita 
# sono simulati con il pacchetto dreno assunto distribuito in corrispondenza della topografia. In questo modo
# si simulano le seepage faces in corrispondenza dei punti di emersione della falda (non noti a priori)
# Bisogna dare in input il DTM della zona più un raster che delinea il bacino e identifica le celle
# a cui assegnare i dreni (in questo caso tutte). 
# # la griglia di calcolo ha layers orizzontali e i dreni vengono assegnati alla quota della topografia
# ==
# questa versione usa una discretizzazione a parallelepipedo e le celle fuori dal bacino (in pianta)
# vengono disattivate(regional_groundwater_model_MIO invece ha ladiscretizzazione che segue la topografia)
#
#Il modluo finale fa anche il particle tracking delle particelle. Le particelle sono distribuite in funzione della ricarica efficace (più dense dove c'è più ricarica)
#Calcola poi il tempo di residenza delle particelle e l'età dello streamflow. Quest'ultimo corrisponde alle età delle 
#particelle che emergono in corrispondenza del pc, pesate in base ai flussi locali delle celle di emersione 

# questa è la versione NON parallelizzata

# itmuni nel dis package definisce le unità di misura del tempo; lenuni definisce quelle spaziali


import flopy
import os
import sys
import flopy.utils.binaryfile as bf
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

#############################################################################
########### Parametri su cui iterare le simulazioni##########################
#############################################################################

#nota: i valori di R corrispondono a ricarica adimensionale R/K pari a [3.17098E-05, 0.000100275,	0.000317098,	0.001002752,	0.003170979,	0.010027517,	0.031709792,	0.100275167,	0.31709792, 1.002751668] 
# nel caso in cui si lasci fissa la conducibilità idraulica (mean_y=-16.12 = mean_k=1e-7m/s)


IMP_DEPTH =[100]                   #[0, 10, 100, 1000]                     # profondità dello strato impermeabilie rispetto alla cella più depressa del dem
R =[	0.679422524] #  logscale: [0.0031536,	0.006794225,	0.014637715,	0.031536,	0.067942252,	0.146377145,	0.31536,	0.679422524,	1.463771455,	3.1536]     #logscale: [0.0005, 0.0014, 0.0043, 0.0130, 0.0389, 0.1168, 0.3504, 1.0512, 3.1536]  # R: [0.02, 0.06325, 0.2, 0.6325, 2]             # ricarica
MEAN_Y = [-16.12]                   #[-18.42, -17.27,  -16.12, -14.97 , -13.82, -12.67,  -11.51]                   #[-18.42, -16.12, -13.82, -11.51]      # media del campo log(k)
VAR_Y = [0]                         # [0, 0, 0, 0, 0, 0, 0]                 # varianza del campo log(k)
TOPOGRAPHY_FACTOR = [0.25]             #[0.04,0.25, 1, 4]                      # parametro che moltiplica le quote del dem per generare topografie più o meno marcate
ALPHA = [0.001]                    #[0, 0.0001, 0.001, 0.01]               # parametro che controlla la decrescita esponenziale della Ks
    
porosity=0.1                          
                   
layer_number=100                    # number of layers (se si cambia bisonga cambiare anche nel generatore dei campi random)

num_particle_per_cell_max=4
R_particle_split=0.679422524        # valore limite della ricarica in base a cui si assgnano al massimo 2 particelle per cella (<) o 4 (>)
  
###########################################################################
#definisce le directories utili############################################
###########################################################################

modelname = "model2" 
modelpath = "../Model_mio/"

exeMODFLOW = "C:/DEV/Exe/MODFLOW-NWT_64.exe"
exeMODPATH = "C:/DEV/Exe/mpath7.exe"

#exeMODFLOW = "../Exe/MODFLOW-NWT_64.exe"
#exeMODPATH = "../Exe/mpath7.exe"

##########################################################################
#open and read raster files ##############################################
##########################################################################

#input necessari: due raster (georeferenziati).  1. DTM; 2. file binario che definisce l'estensione del bacino

#Raster paths
demPath = "../Rst/DEM_Maso_100m.tif"           
crPath =  "../Rst/CR_Maso_100m.tif"            


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

catchment_area= np.sum(crData>0) * geot[1] * abs(geot[5]) / 1000**2  # area bacino in km2
##############################################################################

run_count=0

# alloca alcuni array
drain_fluxes_2D_ARRAY=np.zeros( (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA),) + demData.shape, dtype=np.float32)  # crea array vuoti che verranno riempiti durante i loop sui parametri 
drain_fluxes_minus_recharge_2D_ARRAY=np.zeros( (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA),) + demData.shape, dtype=np.float32)# array dei flussi meno la ricarica - sarebbe la ricarica effettiva
drain_fluxes_directrunoff_2D_ARRAY=np.zeros( (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA),) + demData.shape, dtype=np.float32)# array della componente di runoff (i.e. la componente della ricarica che non si infiltra)

R_K_ratio=np.zeros( ((1),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
Q_gw_Q_sw_ratio=np.zeros( ((1),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
Q_tot=np.zeros( ((1),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
Q_gw_fluxes_normalized=np.zeros( ((1),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
Q_sw_fluxes_normalized=np.zeros( ((1),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
GW_volume=np.zeros( ((1),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)

max_particle_nubmer = num_particle_per_cell_max * sum(sum(crData>0))                    # assegna una particella per ognuna delle celle attive definite da crData - corrisponde al numero massimo (potrebbero essere meno perchè non vengono assegnate particelle se h > piano campagna, vedere sotto) - se si assegnano più particelle per cella bisogna cambiare
traveltime_ARRAY=-9999*np.ones( ((max_particle_nubmer),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
streamflow_age_ARRAY=-9999*np.ones( ((max_particle_nubmer),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)
flowpath_lengths_ARRAY=-9999*np.ones( ((max_particle_nubmer),+ (len(R)*len(MEAN_Y)*len(IMP_DEPTH)*len(TOPOGRAPHY_FACTOR)*len(ALPHA))), dtype=np.float32)        

hist_gw_age_natural=[]
hist_gw_age_log=[]

layer_pc=np.zeros((crData.shape[0], crData.shape[1]),dtype=np.int32)  # layer in cui è contenuto il pc 



for iter_1 in range(len(IMP_DEPTH)):
    
 for iter_2 in range(len(TOPOGRAPHY_FACTOR)): 

  for iter_3 in range(len(ALPHA)):   

   for iter_4 in range(len(MEAN_Y)): 
           
    for iter_5 in range(len(R)):     
       

        ###########################################################################################
        #Initialize Modflow Nwt solver (loop dependent, altrimenti metterlo sopra ai loops)########
        ###########################################################################################
        
        mf1 = flopy.modflow.Modflow(modelname, exe_name= exeMODFLOW, version="mfnwt", model_ws=modelpath)
        nwt = flopy.modflow.ModflowNwt(mf1 , maxiterout=15000,  maxitinner=10000, mxiterxmd = 10000, headtol=0.001, fluxtol=R[iter_5]/50*3600*24, linmeth=1, stoptol=1e-10, hclosexmd =1e-3, dbdtheta = 0.5, backflag=1, msdr=25, thickfact=1e-04)


        ##########################################################################  
        # spatial discretization##################################################
        ##########################################################################
        
        #Boundaries for Dis : Create discretization object, spatial/temporal discretization
        nrow = demDs.RasterYSize
        ncol = demDs.RasterXSize
        delr = geot[1]
        delc = abs(geot[5])       
        
        demData_stretched= TOPOGRAPHY_FACTOR[iter_2] * demData + 1
        crData[crData<0]=0
        demData_stretched[demData_stretched<0]=0
        
        
        ztop = np.ones((nrow , ncol)) *   np.max(demData_stretched[crData>0])       #<------------
        zbot = np.ones((nrow , ncol)) *   np.min(demData_stretched[crData>0]) - IMP_DEPTH[iter_1]
        nlay = layer_number
        delv = (ztop - zbot) / nlay
        botm = np.linspace(ztop, zbot, nlay + 1)
                 
        ################################################################################
        #definition of flow packages ###################################################
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
        
            f = open('field.dat', 'rb')
            dt = np.dtype('f8')
            hk = np.fromfile( f, dtype=dt, count=-1 ).reshape((nlay,nrow,ncol),order='C')
            f.close()
        
        hk= np.exp(np.sqrt(VAR_Y[iter_4])* hk + MEAN_Y[iter_4])  # passa dalla log conductivity normalizzata alla conducibilità vera e propria  <----------------
        
        hk= hk*3600*24  # passa a m/giorno
        
        #########################################################################
        # fa variare la conducibilità idraulica con la profondità (la matrice reduction_factor_Ks corrisponde a come le k vengono scalate, decresce esponenzialmente da 1 a un valore minimo ora impostato a 0.01)
        #######################################################################
        reduction_factor_Ks = np.ones(hk.shape, dtype=np.float32)
        for idx1 in range(nrow):    
         for idx2 in range(ncol):
          for idx3 in range(nlay):   
           if demData_stretched[idx1,idx2] >= botm[idx3+1,idx1,idx2] and demData_stretched[idx1,idx2]>0:    
            reduction_factor_Ks[idx3,idx1,idx2] = 0.05 + (1-0.05) * np.exp(- ALPHA[iter_3] * (demData_stretched[idx1,idx2] - ( botm[idx3,idx1,idx2] + botm[idx3+1,idx1,idx2] )/2  )  )
        
        hk= np.multiply(hk,reduction_factor_Ks)
        
        ######################################################################################## 

        laytyp=np.ones((nlay), dtype=int)
        
        # Variables for the DIS package
        dis = flopy.modflow.ModflowDis(mf1, nlay,nrow,ncol,delr=delr,delc=delc,top=ztop,botm=botm[1:],itmuni=4) # <------itmuni=1: secondi; itmuni = 4: giorni 
        
        # Variables for the BAS package
        iboundData = np.zeros(demData.shape, dtype=np.int32)
        iboundData[crData > 0 ] = 1
        
        
        #condizini iniziali di primo tentativo
        
        #strt= zbot + IMP_DEPTH[iter_1] + (ztop - zbot - IMP_DEPTH[iter_1]) * R[iter_5]/365/86400 / np.mean(hk) 
        #strt= demData_stretched * R[iter_5]/365/86400 / np.mean(hk) + 1
        #strt= demData_stretched * 0.5 + 200
        strt= demData_stretched * 0.5 +100
        #strt= demData_stretched * 1/(1 + np.exp(5-10*R[iter_5]/365/86400 / np.mean(hk))) + 100
        #strt= zbot + Imp_depth - 10
        #strt= demData_stretched
        
# =============================================================================
#         # usa il potenziale del loop precedente come condizione iniziale - commentare se non si vuole fare cosi
#         if 'head_0' in globals():
#          strt= head_0[:,:]
#         else:
#          strt= demData_stretched * 0.5 + 200
# =============================================================================
        
        ######################################
        # Add BAS package to the MODFLOW model
        ######################################
        bas = flopy.modflow.ModflowBas(mf1,ibound=iboundData,strt=strt, hnoflo=-2.0E+020)    # <-------------
        
        ######################################
        # Add UPW package to the MODFLOW model
        ######################################
        upw = flopy.modflow.ModflowUpw(mf1, laytyp = laytyp, hk = hk, ipakcb=53, hdry = -9999 , iphdry = 1) # <----!!!!! hdry = -1 , iphdry = 1)   #  <---- IMPORTANTE! definisce come vengono gestite le celle che si asciugano perchè la falda scende 
        
           
        ################################################################
        #Add the recharge package (RCH) to the MODFLOW model###########
        ################################################################
 
        rch_array = np.zeros((nrow, ncol), dtype=np.float32)     
        rch_array[crData>0]=R[iter_5]/365  #/3600/24    #m/giorno
       
        # prova - non viene assegnata la ricarica alle celle in cui h > topografia nello step di R precedente o altre condizioni
# =============================================================================
#         if iter_5>0:
#             rch_array[drain_fluxes_2D<0]=0
#             #rch_array[head_0[:,:]>demData_stretched]=0
#             rch_array[rch_array_old==0]=0
#         
#         rch_array_old=rch_array
# =============================================================================
        
        rch_data = {0: rch_array}
        rch = flopy.modflow.ModflowRch(mf1, nrchop=3, rech =rch_data)
        
          
        # =============================================================================
        # #Add the evapotranspiration package (EVT) to the MODFLOW model
        # evtr = np.ones((nrow, ncol), dtype=np.float32) * 0.1/3600/24/365
        # evtr_data = {0: evtr}
        # evt = flopy.modflow.ModflowEvt(mf1,nevtop=1,surf=ztop,evtr=evtr_data, exdp=0.5)
        # =============================================================================
        
        
        
        #################################################################################
        #Add the drain package (DRN) to the MODFLOW model################################
        #################################################################################
                
        sorgenti = np.zeros(demData.shape, dtype=np.int32)
        sorgenti[crData >0 ] = 1
        lista = []
        for i in range(sorgenti.shape[0]):
            for q in range(sorgenti.shape[1]):
                
                for j in range(nlay):
                 if   demData_stretched[i,q] < botm[j,i,q] and demData_stretched[i,q] > botm[j+1,i,q] and sorgenti[i,q]>0: 
                  w=j
                  layer_pc[i,q]=j  #assegna ad ogni cella il laryer in cui cade il piano campagna
                
                if sorgenti[i,q] == 1:
                    #C=hk[w,i,q]*delc*delr/0.1
                    C=1 *3600*24  # diviso per 1e-8 nel caso in cui si usasse la ricarica adimensionale
                    lista.append([w,i,q,demData_stretched[i,q], C ]) #layer,row,column,elevation(float),conductance  <--------------  
        rivDrn = {0:lista}
        
        drn = flopy.modflow.ModflowDrn(mf1,ipakcb=53, stress_period_data=rivDrn, filenames=None)
        
        
        # Add OC package to the MODFLOW model
        
        #spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
        #oc = flopy.modflow.ModflowOc(mf1, stress_period_data=spd, compact=True)
       
        #oc = flopy.modflow.ModflowOc(mf1)
        
        flopy.modflow.ModflowOc(mf1, stress_period_data={(0, 0): ['save head','save budget', 'print head']})
                
        ########################################################################
        ####### writes files and run simulation################################
        #######################################################################
        
        #Write input files -> write file with extensions
        mf1.write_input()
        
        #run model -> gives the solution
        mf1.run_model()
    
    
        #########################################################
        #legge il file dei potenziali e dei flussi dai dreni#####
        #########################################################
        
        fname = os.path.join(modelpath, modelname + ".hds")
        hds = bf.HeadFile(fname)
        times = hds.get_times()
        
        fname = os.path.join(modelpath, modelname + ".cbc")
        cbb = bf.CellBudgetFile(fname)
        kstpkper_list = cbb.get_kstpkper()
        
        
        #calcola vari flussi
        drain_fluxes_3D= cbb.get_data(kstpkper=(0,0), text='DRAIN',full3D=True)  #flussi in modflow sono in L^3/T
        drain_fluxes_2D= np.sum(drain_fluxes_3D[0], axis=0)
        drain_fluxes_2D=drain_fluxes_2D/(delc*delr)          # <------- passa da L^3/T a L/T (da portata a flusso)
        drain_fluxes_2D[crData==0]=np.NaN
        drain_fluxes_minus_recharge_2D=drain_fluxes_2D + rch_array 
        
        drain_fluxes_2D_ARRAY[run_count,:,:]=drain_fluxes_2D             
        drain_fluxes_minus_recharge_2D_ARRAY[run_count,:,:]=drain_fluxes_minus_recharge_2D  # flussi in entrata (+) o uscita (-) dal sottosuolo:facendo la differenza  tra flussi dai dreni e RCH è possibile stabilire la componente effettiva della ricarica (infatti quando la falda raggiunge il pc la ricarica effettiva è 0, nonostante quello che le si è assegnato a RCH)

        
        #direct runoff, ovvero la frazione di pioggia che diventa subito runoff perchè non si infiltra (cade su celle che hanno flusso sotterraneo in uscita)
        drain_fluxes_directrunoff_2D=np.copy(drain_fluxes_2D)
        drain_fluxes_directrunoff_2D[drain_fluxes_directrunoff_2D<-R[iter_5]/365]=-R[iter_5]/365  #<<<<<< in secondi: runoff[runoff<-R[iter_5]/365/3600/24]=-R[iter_5]/365/3600/24 -  componente del deflusso che non entra nel groundwater ma viene drenata subito 
        drain_fluxes_directrunoff_2D_ARRAY[run_count,:,:]=drain_fluxes_directrunoff_2D
    
        
        #extracts heads 
        fname = os.path.join(modelpath, 'model2.hds')
        hdobj = flopy.utils.HeadFile(fname)
        head = hdobj.get_data()
        head_0=head[-1,:,:]  #<-------- 
        
        #groundwater volume (km^3)
        GW_volume[0,run_count] = np.sum(head>100) * delc * delr * delv[0,0] * porosity / 1000**3
    
    
        #######################################################################
        #######################################################################
        #### particle tracking with modpath 7
        #######################################################################
        #######################################################################

        ###################################################################################
        # crea particelle distirbuite uniformemente (ricarica)--per il tracking FORWARD
        ###################################################################################
                
        plocs = []
        pids  = []
        localx= []
        localy= []
        localz= []
        particle_count=0

#assegna un numero di particelle proporzionale al flusso netto in entrata nell'acquifero 

        for idx1 in range(nrow):    # le particelle vengono assegnate per ogni cella alla quota corrispondente alla piezomatrica 
         for idx2 in range(ncol):   
           if crData[idx1,idx2] >0 and drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]>0:   #rch_array[idx1,idx2] >0:    # demData_stretched[idx1,idx2] >0 and head_0[idx1,idx2]<=demData_stretched[idx1,idx2]:   # assegna le particelle solo nei punti in cui la falda è effettivamente sotto il pc
              
               R_m_day=R[iter_5]/365
              
               if R[iter_5]>=R_particle_split:
                   
                   num_particle_per_cell_max_case= 4
                   
                   if  (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  >  4/5 * R_m_day) & (crData[idx1,idx2]==1):
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                       #plocs.append((layer_pc[idx1,idx2]+1, idx1, idx2))      # con 0 al primo posto le particelle vengano rilasciae nel primo layer attivo. ora vengono messe in quello successivo per evitare che vengano immediatamente rimosse - necessario perchè seenò si possono avere particelle che vengono immediatamente rimosse nonostante siano in celle di downwelling (ovvero con drain_fluxes_minus_recharge_2D_ARRAY positivo)
                 
                       localx.append(0.25)  # posizioni relative all'interno di ciascuna cella (ora assegna 4 particelle per cella, vedere all'inizio "num_particle_cell!")
                       localx.append(0.25)
                       localx.append(0.75)
                       localx.append(0.75)
                   
                       localy.append(0.25)
                       localy.append(0.75)
                       localy.append(0.25)
                       localy.append(0.75)
                   
                       localz.append(0.95)  # 0 corrisponde al fondo della cella, 1 al top
                       localz.append(0.95)
                       localz.append(0.95)
                       localz.append(0.95)
                   
                       pids.append(particle_count)              
                       particle_count+=1  
                       pids.append(particle_count)
                       particle_count+=1 
                       pids.append(particle_count)              
                       particle_count+=1  
                       pids.append(particle_count)
                       particle_count+=1

                   if  (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  >  3/5 * R_m_day)  &  (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  <  4/5 * R_m_day)  & (crData[idx1,idx2]==1) :
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                 
                       localx.append(0.33)  # posizioni relative all'interno di ciascuna cella (ora assegna 4 particelle per cella, vedere all'inizio "num_particle_cell!")
                       localx.append(0.66)
                       localx.append(0.5)
           
                       localy.append(0.33)
                       localy.append(0.33)
                       localy.append(0.66)
  
                       localz.append(0.95)  # 0 corrisponde al fondo della cella, 1 al top
                       localz.append(0.95)
                       localz.append(0.95)
                   
                       pids.append(particle_count)              
                       particle_count+=1  
                       pids.append(particle_count)
                       particle_count+=1 
                       pids.append(particle_count)              
                       particle_count+=1  

                   if  (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  >  2/5 * R_m_day)  &  (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  <  3/5 * R_m_day)  & (crData[idx1,idx2]==1) :
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                  
                       localx.append(0.33)  # posizioni relative all'interno di ciascuna cella (ora assegna 4 particelle per cella, vedere all'inizio "num_particle_cell!")
                       localx.append(0.66)
           
                       localy.append(0.33)
                       localy.append(0.66)
  
                       localz.append(0.95)  # 0 corrisponde al fondo della cella, 1 al top
                       localz.append(0.95)
                   
                       pids.append(particle_count)              
                       particle_count+=1  
                       pids.append(particle_count)
                       particle_count+=1 
                       
                   if  (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  >  1/5 * R_m_day)  & ( drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  <  2/5 * R_m_day)  & (crData[idx1,idx2]==1):
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                  
                       localx.append(0.5)  # posizioni relative all'interno di ciascuna cella (ora assegna 4 particelle per cella, vedere all'inizio "num_particle_cell!")
           
                       localy.append(0.5)
  
                       localz.append(0.95)  # 0 corrisponde al fondo della cella, 1 al top
                   
                       pids.append(particle_count)              
                       particle_count+=1                    
                       
                       
                       
                       
               elif R[iter_5]<R_particle_split:
                   
                   num_particle_per_cell_max_case= 2
                   
                   if  (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  >  2/3 * R_m_day)  & (crData[idx1,idx2]==1):
                       
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
    
                       #plocs.append((layer_pc[idx1,idx2]+1, idx1, idx2))      # con 0 al primo posto le particelle vengano rilasciae nel primo layer attivo. ora vengono messe in quello successivo per evitare che vengano immediatamente rimosse - necessario perchè seenò si possono avere particelle che vengono immediatamente rimosse nonostante siano in celle di downwelling (ovvero con drain_fluxes_minus_recharge_2D_ARRAY positivo)
                       #plocs.append((layer_pc[idx1,idx2]+1, idx1, idx2))
                       #plocs.append((layer_pc[idx1,idx2]+1, idx1, idx2))
                  
                       localx.append(0.25)  # posizioni relative all'interno di ciascuna cella (ora assegna 4 particelle per cella, vedere all'inizio "num_particle_cell!")
                       localx.append(0.75)
    
                       localy.append(0.25)
                       localy.append(0.75)
        
                       localz.append(0.95)  # 0 corrisponde al fondo della cella, 1 al top
                       localz.append(0.95)
                   
                       pids.append(particle_count)              
                       particle_count+=1  
                       pids.append(particle_count)
                       particle_count+=1 

                   if  (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  >  1/3 * R_m_day)  &   (drain_fluxes_minus_recharge_2D_ARRAY[run_count,idx1,idx2]  <  2/3 * R_m_day) & (crData[idx1,idx2]==1):
                       
                       plocs.append((layer_pc[idx1, idx2], idx1, idx2))
                  
                       localx.append(0.5)  # posizioni relative all'interno di ciascuna cella (ora assegna 4 particelle per cella, vedere all'inizio "num_particle_cell!")
    
                       localy.append(0.5)
        
                       localz.append(0.95)  # 0 corrisponde al fondo della cella, 1 al top
                   
                       pids.append(particle_count)              
                       particle_count+=1  


           
        part0 = flopy.modpath.ParticleData(plocs, drape=1, structured=True, particleids=pids, localx=localx, localy=localy, localz=localz)  # drape=1: mette le particelle nel primo layer attivo sotto quello specificato in "plocs". drape=0 mette le particelle nel layer specificato in "plocs", se il layer non è attivo le particelle vengono eliminate
        pg0 = flopy.modpath.ParticleGroup(particlegroupname='PG1', particledata=part0,filename='ex01a.pg1.sloc')
        

        
       ##################################################################################### 
       # create particles distirbuite uniformemente (ricarica)--per il tracking BACKWARD#### 
       #####################################################################################
        
    # =============================================================================
    #     temp=np.where(drain_fluxes_3D[0].data<0)   # identifica celle di flusso uscente   
    #    
    #     plocs = []
    #     pids = []
    #     particle_count=0
    #     
    #     for i in range(len(temp[0])):
    #         plocs.append((temp[0][i], temp[1][i], temp[2][i]))
    #         pids.append(particle_count)
    #         particle_count=particle_count+1
    #         
    #     part0 = flopy.modpath.ParticleData(plocs, drape=1, structured=True, particleids=pids)
    #     pg0 = flopy.modpath.ParticleGroup(particlegroupname='PG1', particledata=part0,filename='ex01a.pg1.sloc')
    # =============================================================================


        
        particlegroups = [pg0]
        
        # default iface for MODFLOW-2005 and MODFLOW 6
        defaultiface = {'RECHARGE': 6, 'ET': 6, 'DRN': 6}     #<---- definisce dove le condizioni vengono applicate (0: cella intera, 1-5: facce laterali e inferiore, 6:faccia superiore)
        defaultiface6 = {'RCH': 6, 'EVT': 6, 'DRN': 6}
        
        
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
        

        ##########################################################################
        # get pathline file (file molto voluminoso, commentare se non necessario)
        ##########################################################################
        
        #import flopy.utils.modpathfile as mpf                             
        pthobj = flopy.utils.PathlineFile(modelpath + modelname + '_mp'+'.mppth')
        p = pthobj.get_alldata()          # pathfile per tutte le particelle
        #p1 = pthobj.get_data(partid=1)   # pathfile per una particlella specificata
        
        #conta le particelle che si infiltrano (GW) e quelle che sono state escluse (SW)
        num_particles_GW= len(p)
        num_particles_SW= np.sum(crData>0) * num_particle_per_cell_max_case  - num_particles_GW 
        
        # Calcola la lunghezza dei flowpaths e velocità delle particelle a partire dall'oggetto pathfile
        flowpath_lengths=np.zeros( (num_particles_GW,), dtype=np.float32)
        flowpath_age_mean=np.zeros( (num_particles_GW,), dtype=np.float32)
        flowpath_velocities=[]
        flowpath_velocities_mean=[]
        
        gw_age=[]
        gw_age_weights=[]


        for i in range(num_particles_GW):                                                         # itera su tutte le particelle
            coord_array=np.array([p[i].x[:-1], p[i].y[:-1], p[i].z[:-1], p[i].time[:-1]]).T                 # crea array nx3 con i vertici della traiettoria di un flowpat
            delta_length = np.sum(np.sqrt(np.diff(coord_array[:,0:3], axis=0)**2), axis=1)    # calcola lunghezza singolo flowpath
            flowpath_lengths[i] = np.sum(delta_length)
            delta_time=np.diff(coord_array[:,3], axis=0)
            flowpath_velocities.append(delta_length/delta_time)
            
            #non sono sicuro sia il modo giusto di calcolare la velocità media (media temporale vs media spaziale !!!)
            #flowpath_velocities_mean.append(  np.mean(flowpath_velocities[-1][np.isfinite(flowpath_velocities[-1])])   )
            #flowpath_velocities_mean.append( np.sum( flowpath_velocities[-1][np.isfinite(flowpath_velocities[-1])]  * delta_length[np.isfinite(flowpath_velocities[-1])]  )  /  flowpath_lengths[i]  )  #velocità media del singolo percorso di flusso
            
            flowpath_age_mean[i]= np.sum(  (coord_array[0:-1,3] * delta_length / flowpath_velocities[-1])  [np.isfinite(1/flowpath_velocities[-1])]   )    /   np.sum(  1/ flowpath_velocities[-1][np.isfinite(1/flowpath_velocities[-1])] * delta_length[np.isfinite(1/flowpath_velocities[-1])]  )
        
            #l'età del GW è l'età di ogni particella in diverse posizioni pesata in funzione del delta_length  e dell'area del tubo di flusso (proporzionale all'inverso della velocità). Si assume che ongi tubo di flusso corrisponda alla stessa portata
            gw_age.append( coord_array[0:-1,3][np.isfinite(1/flowpath_velocities[-1])]  )
            gw_age_weights.append( (delta_length / flowpath_velocities[-1]) [np.isfinite(1/flowpath_velocities[-1])]  )
 
        #flowpath_velocities_mean=np.array(flowpath_velocities_mean)    
        flowpath_lengths_ARRAY[0:len(flowpath_lengths),run_count]=flowpath_lengths           
 
    
        #calcola statistiche della distribuzione delle età del gw   
     
        #crea un vettore con tutte le età delle particelle    
        if len(gw_age)>0:
            gw_age_array=np.array([])
            gw_age_weights_array=np.array([])
            for i in range(len(gw_age)): 
                gw_age_array= np.append(gw_age_array, gw_age[i]) 
                gw_age_weights_array= np.append(gw_age_weights_array, gw_age_weights[i]) 
            
            def weighted_mean(var, wts):
                 return np.average(var, weights=wts)
            def weighted_variance(var, wts):
                 return np.average((var - weighted_mean(var, wts))**2, weights=wts)
            def weighted_skew(var, wts):
                 return (np.average((var - weighted_mean(var, wts))**3, weights=wts) / weighted_variance(var, wts)**(1.5))
            def weighted_kurtosis(var, wts):
                 return (np.average((var - weighted_mean(var, wts))**4, weights=wts) / weighted_variance(var, wts)**(2))
            
            gw_age_mean=weighted_mean(gw_age_array,gw_age_weights_array)
            gw_age_var=weighted_variance(gw_age_array,gw_age_weights_array)
            gw_age_skew=weighted_skew(gw_age_array,gw_age_weights_array)
            gw_age_kurt=weighted_kurtosis(gw_age_array,gw_age_weights_array)
            
            #computes histogram characteristics
            hist, bins, _ = plt.hist(gw_age_array, bins=25, range= (10, np.max(gw_age_array)),density=True, weights=gw_age_weights_array)
            logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            hist_log, bins_log, _ =plt.hist(gw_age_array, bins=logbins, density=True, weights=gw_age_weights_array)
            bins_width=bins[1:]-bins[0:-1]
            bins_width_log=bins_log[1:]-bins_log[0:-1]
            hist_gw_age_natural.append([hist,bins[:-1],bins_width])
            hist_gw_age_log.append([hist_log,logbins[:-1],bins_width_log])



        Q_gw_particle= R[iter_5] * delr * delc / num_particle_per_cell_max_case  *  num_particles_GW  /365/24/3600  # m3/sec
        Q_sw_particle= R[iter_5] * delr * delc / num_particle_per_cell_max_case  *  num_particles_SW  /365/24/3600  # m3/sec
        Q_gw_Q_sw_ratio_particle=Q_gw_particle/Q_sw_particle
              
        Q_gw_fluxes= np.nansum(drain_fluxes_minus_recharge_2D[drain_fluxes_minus_recharge_2D>0]) * delr * delc/24/3600
        Q_sw_fluxes= -1*np.nansum(drain_fluxes_directrunoff_2D) * delr * delc /24/3600
        Q_gw_Q_sw_ratio_fluxes=Q_gw_fluxes/Q_sw_fluxes
        
        Q_tot[0,run_count]=R[iter_5] /365/24/3600 * catchment_area *1000**2
        Q_gw_fluxes_normalized[0,run_count]=Q_gw_fluxes/Q_tot
        Q_sw_fluxes_normalized[0,run_count]=Q_sw_fluxes/Q_tot
  
        Q_gw_Q_sw_ratio[0,run_count]=Q_gw_Q_sw_ratio_fluxes
        
        ###############################
        # get travel times
        ###############################
        
        endobj = flopy.utils.EndpointFile(modelpath + modelname + '_mp'+'.mpend')   # ottiene tutti i travel times
        e = endobj.get_alldata()
        traveltime=e.time
        e1 = endobj.get_data(partid=1)   # travel time per una particlella specificata
        
        
        traveltime_ARRAY[0:len(traveltime),run_count]=traveltime
        
        
        
        
        #####################################################################################################################
        ###### estrae le traveltimes in corrispondenza dei punti in cui i dreni sono attivi e calcola l'età dello streamflow pesando l'età delle particelle in uscita per il loro flusso
        #####################################################################################################################
        
        endobj = flopy.utils.EndpointFile(modelpath + modelname + '_mp'+'.mpend')  # estra l'endobject
        outflow_kij=np.where(drain_fluxes_3D[0].data < -R[iter_5]/365*delc*delr)   #3600/24    # identifica le celle in cui vi sono flussi emergenti (in questo caso i dreni estraggono quello che viene da sotto più la ricarica che viene subito drenata)
    
        flux=[]
        flux_minus_recharge=[]
        time_particles_at_cell=[]
        streamflow_age=[]
        streamflow_age_weight=[]
        
        #per ogni cella in cui ci sono flussi in uscita estra le età delle particelle in uscita
        for i in range(len(outflow_kij[0])): 
            flux.append(drain_fluxes_3D[0].data[outflow_kij[0][i], outflow_kij[1][i], outflow_kij[2][i]]) 
            e0 = endobj.get_destination_endpoint_data(dis.get_node((outflow_kij[0][i], outflow_kij[1][i], outflow_kij[2][i])))  # dis.get_node ottiene a partire dalla terna k,i,j il codice identificativo univoco della cella 
            time_particles_at_cell.append(e0.time) 
    
    
# =============================================================================
#         for i in range(len(flux)):  #    rimuove dai flussi in usicta la componente dovuta alla ricarica che cade in corrispondenza dei punti da cui i dreni estraggono 
#             flux_minus_recharge.append(flux[i]  + R[iter_5]/3600/24/365*delc*delr)
#     
#         
#         flux_array = np.array(flux_minus_recharge)
#         
#         flux_sub   = np.zeros(np.shape(flux_array),dtype=float)
#         flux_sup   = np.zeros(np.shape(flux_array),dtype=float)
#         
#         flux_sub[flux_array<0] = flux_array[flux_array<0]  #flussi_sub: sono i flussi che emergono dal sottosuolo
#         flux_sup[flux_array>=0]= flux_array[flux_array>=0] # flussi sup: sono la componente della ricarica che non si infilta (casi in cui non tutta la ricarica riesce ad infiltrarsi o casi in cui la ricarica è assegnata a punti in cui ci sono flussi sotterranei in uscita)
#         flux_sup[flux_array<0] = R[iter_5]/3600/24/365*delc*delr
#         
#         
#         flux_sub_sum   = np.abs(np.sum(flux_sub))           #componente di groundwater (fuoriesce dopo essersi infiltrata)
#         flux_sup_sum   = np.abs(np.sum(flux_sup))           #componente di acqua superficiale (ricarica che non si infiltra e diveta subito runoff superficiale). equivalente a: np.sum(crData>0)*delc*delr*R[iter_5]/3600/24/365 - flux_sub_sum
#     
#     
#     
#        # tentativo parte  nuova
#         for j in range(len(time_particles_at_cell)):    #calcola le età pesate con i flussi
#          for jj in range(len(time_particles_at_cell[j])):
#             streamflow_age.append( time_particles_at_cell[j][jj] )
#             streamflow_age_weight.append(  -  flux_minus_recharge[j] / (flux_sub_sum + flux_sup_sum) )
#         
#         streamflow_age=np.log10(np.array(streamflow_age))    # passa da liste ad array e trasforma le età in log (età)
#         streamflow_age_weight=np.array(streamflow_age_weight)
#         
#         [streamflow_age_hist_values, streamflow_age_hist_bins] = np.histogram(streamflow_age, weights=streamflow_age_weight,bins=50, range=(1,15)) # istogramma pesato sui flussi. da in output i bins delle età e le frequenze pesate (i.e. età dellgli streamflows)
#     
# =============================================================================
    
    # parte vecchi, probabilmente sbagliata
# =============================================================================
#         for j in range(len(time_particles_at_cell)):    #calcola le età pesate con i flussi
#          for jj in range(len(time_particles_at_cell[j])):
#             streamflow_age.append(time_particles_at_cell[j][jj]*flux[j] / sum(flux) )
# =============================================================================
            
        #streamflow_age=np.array(streamflow_age)     
            
            
        #streamflow_age_ARRAY[0:len(streamflow_age),run_count]=streamflow_age
    
        R_K_ratio[0,run_count] = ( R[iter_5] / 365) /  np.exp(  (MEAN_Y[iter_4] + VAR_Y[iter_4])/2)   #<<<<<<3600/24
    
        #########################################################################################
        ####### end particle tracking ###########################################################
        #########################################################################################
        
        print(run_count)
        run_count=run_count+1
     
        
##################################################################################
#######end model runs#############################################################
##################################################################################





#######################################################################
## model results and post processing###################################
#######################################################################


#Import model
ml = flopy.modflow.Modflow.load('../Model_mio/model2.nam')    #<---------

# First step is to set up the plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')

# Next we create an instance of the ModelMap class
modelmap = flopy.plot.ModelMap(sr=ml.dis.sr)

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = modelmap.plot_grid(linewidth=0.4)



#Cross section of model grid representation

fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 1, 1)
# Next we create an instance of the ModelCrossSection class
#modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Column': 5})
modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Row': 99})

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = modelxsect.plot_grid(linewidth=0.4)
t = ax.set_title('Column 6 Cross-Section - Model Grid')



#Active/inactive cells on model extension

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=ml, rotation=0)
quadmesh = modelmap.plot_ibound(color_noflow='cyan')
linecollection = modelmap.plot_grid(linewidth=0.4)


#Cross sections of active/inactive cells

fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 1, 1)
modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Column': 5})
patches = modelxsect.plot_ibound(color_noflow='cyan')
linecollection = modelxsect.plot_grid(linewidth=0.4)
t = ax.set_title('Column 6 Cross-Section with IBOUND Boundary Conditions')


#Channel network as drain (DRN) package

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=ml, rotation=-14)
quadmesh = modelmap.plot_ibound(color_noflow='cyan')
quadmesh = modelmap.plot_bc('DRN', color='blue')
linecollection = modelmap.plot_grid(linewidth=0.4)


#Model grid and heads representation

fname = os.path.join(modelpath, 'model2.hds')
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()

fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(1, 2, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=ml, rotation=-14)
#quadmesh = modelmap.plot_ibound()
quadmesh = modelmap.plot_array(head, masked_values=[-2.e+20], alpha=0.8)
linecollection = modelmap.plot_grid(linewidth=0.2)

#############################################################
#Plot fluxes out of drains 3D (fette)#######################
############################################################

a=drain_fluxes_3D[0]
a=a.data[40,:,:]
fig=plt.imshow(a, interpolation='none',vmin=-0.005, vmax=0)
plt.show()

#############################################################################
#Plot fluxes out of drains 2D (somma lungo le verticali)#####################
#############################################################################

##plot singolo

fig=plt.imshow(drain_fluxes_2D, interpolation='none')  # ...interpolation='none',vmin=-0.01, vmax=0
plt.show()

####################
#plotta distribuzion
####################

#traveltime (i.e. età della componente dello streamflow formata da flussi sotterranei)
scenario = 0  # quale risultato plottare
data=traveltime_ARRAY[traveltime_ARRAY[:,scenario]>0,scenario]

plt.subplot(211)
hist, bins, _ = plt.hist(data, bins=25, range= (10, np.max(traveltime_ARRAY)))

# histogram on log scale. 
# Use non-equal bin sizes, such that they look equal on log scale.
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.subplot(212)
plt.hist(data, bins=logbins, density=True)
plt.xscale('log')
plt.show()


#età del groundwater come istrogramma calcolato a partire dalle frequenze pesate
plt.subplot(211)
hist, bins, _ = plt.hist(gw_age_array, bins=25, range= (10, np.max(gw_age_array)), density=True, weights=gw_age_weights_array)

# histogram on log scale. 
# Use non-equal bin sizes, such that they look equal on log scale.
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.subplot(212)
plt.hist(gw_age_array, bins=logbins, density=True, weights=gw_age_weights_array)
plt.xscale('log')
plt.show()


#età delgroundwater come grafico a barre a partire dalle dimensioni in hist_gw_age
scenario = 0 # quale risultato plottare
plt.subplot(211)
plt.bar(hist_gw_age_natural[scenario][1], hist_gw_age_natural[scenario][0], align='edge', width=hist_gw_age_natural[scenario][2])
plt.subplot(212)
plt.bar(hist_gw_age_log[scenario][1], hist_gw_age_log[scenario][0], align='edge', width=hist_gw_age_log[scenario][2])
plt.xscale('log')
plt.show()
##plotta tutti

# =============================================================================
# plt.figure()
# f, axarr = plt.subplots(2,5) 
# 
# axarr[0,0].imshow(drain_fluxes_2D_ARRAY[0,:,:], interpolation='none')  
# axarr[0,1].imshow(drain_fluxes_2D_ARRAY[1,:,:], interpolation='none')  
# axarr[0,2].imshow(drain_fluxes_2D_ARRAY[2,:,:], interpolation='none')  
# axarr[0,3].imshow(drain_fluxes_2D_ARRAY[3,:,:], interpolation='none')  
# axarr[0,4].imshow(drain_fluxes_2D_ARRAY[4,:,:], interpolation='none')  
# axarr[1,0].imshow(drain_fluxes_2D_ARRAY[5,:,:], interpolation='none')  
# axarr[1,1].imshow(drain_fluxes_2D_ARRAY[6,:,:], interpolation='none')  
# axarr[1,2].imshow(drain_fluxes_2D_ARRAY[7,:,:], interpolation='none')  
# axarr[1,3].imshow(drain_fluxes_2D_ARRAY[8,:,:], interpolation='none')  
# axarr[1,4].imshow(drain_fluxes_2D_ARRAY[9,:,:], interpolation='none')
# 
# 
# =============================================================================


############################################
##### saves variables
############################################

# =============================================================================
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
# =============================================================================


############################################
##### load variables
############################################

# =============================================================================
# import shelve
# 
# filename='C:/PostDoc/Acquiferi_trentini/Risultati/AAA_adimensionali_full_run/shelve.out'
# 
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()
# =============================================================================
