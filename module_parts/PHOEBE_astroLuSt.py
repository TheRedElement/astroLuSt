
    ###################
    #Steinwender Lukas#
    ###################

#______________________________________________________________________________
#Class useful for working with PHOEBE
#TODO: get extract_data(), plot_data_PHOEBE inside this class
class PHOEBE_additional:

    def __init__(self):
        pass

#______________________________________________________________________________
#function to extract data from phoebe b. object
def extract_data(phoebes_b, distance, unit_distance, show_keys=False, timeit=False):
    """
    Function to extract data from PHOEBE bundle

    Parameters
    ----------
    phoebes_b : PHOEBEs default binary object
        data structure object of PHOEBE to be converted to dict.
    distance : float
        distance to observed object. If not known simply type 1
    unit_distance : str
        a string to specify the unit of distance.
    show_keys : bool, optional
        If true prints out all possible keys (twigs) of the created dictionary. The default is False.
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        

    Returns
    -------
    dictionary of PHOEBEs default binary object with all the data-arrays in it.
    The structure is the following:
    binary = {
            "lc-dataset1":{
                "enabled":True
                "model1":{"qualifier1":np.array([values]), "qualifier2":np.array([...]), ... ,"qualifieri":np.array([...])},
                "model2":{...},
                :
                :
                :
                "modeli":{...}
                }, 
            "lc-dataset2":{"enabled":True, qualifier1:np.array([values]),...},
            :
            :
            :
            "lc-dataseti":{...},
            "orb-daraset1:{
                "model1":{
                    "component1":{"qualifier1":np.array([values]), "qualifier2":np.array([...]), ... ,"qualifieri":np.array([...])}
                    "component2":{"qualifier1":np.array([values]), "qualifier2":np.array([...]), ... ,"qualifieri":np.array([...])}
                    :
                    :
                    "componenti":{"qualifier1":np.array([values]), "qualifier2":np.array([...]), ... ,"qualifieri":np.array([...])}
                    },
                "model2":
                :
                :
                "modeli":
                }
        }
    
    Notes
    -----
    to add new dataset: add another if d[...:...] == "newdataset", and extract the date with phoebes twig-system
        

    """
    
    import numpy as np
    from utility_astroLuSt import Time_stuff
    
    #time execution
    if timeit:
        task = Time_stuff("extract_data")
        task.start_task()
    
    
    #convert given distance to Solar Radii, to have correct units            
    if distance == 0:
        raise ValueError("distance can not be 0!!! Choose a estimate value for the distance to the object!!\nValue has te be entered in meters!!")

    
    R_Sun_m = 696342*10**3 #m
    R_Sun_Lj = 7.36*10**-8 #Lj
    R_Sun_pc = 2.25767*10**-8 #pc


    if unit_distance == "m":
        distance /= R_Sun_m
    elif unit_distance == "Lj":
        distance /= R_Sun_Lj
    elif unit_distance == "pc":
        distance /= R_Sun_pc
    elif unit_distance == "Sol_Rad":
        distance /= 1
    else:
        print("unit of distance has to be a string of: m, Lj, pc, Sol_Rad!!")
        return

    
    
    b = phoebes_b
    binary = {}     #create dictionary to store values according to phoebe twigs
    
    #extracts the given data of specified datasets
    #to add a new dataset add another "if d[... : ...] == "newdataset" ", and extract the date with phoebes twig-system
    for d in b.datasets:
        binary[d] = {}
        binary[d]["enabled"] = b.get_value(dataset=d, qualifier="enabled")
        binary[d]["binary"] = {}
#        print("______________")
#        print("\ndataset: %s"%(d))
        
        #only extract data saved under componet=binary, if enabled == True
        if binary[d]["enabled"]:
            for q in b[d]["binary"].qualifiers:
                binary[d]["binary"][q] = b.get_value(dataset=d, component="binary", qualifier=q)
#                print("dataset: %s, component: binary, qualifier: %s" %(d, q))
        
        for m in b[d].models:
            binary[d][m] = {}
#            print("dataset: %s, model: %s"%(d, m))
            
            #extract all available orbit datasets of the created models
            if d[0:3] == "orb":
                for c in b[d][m].components:
                    if c != "binary":
                        binary[d][m][c] = {}
                        binary[d][m][c]["phases"] = b.to_phase(b.get_value(dataset=d, model=m, component=c, qualifier="times"))
#                        print("dataset: %s, model: %s, component: %s"%(d,m,c))

#TODO: NOT QUITE SURE IF CONVERSION to [''] IS CORRECT
                        us = b.get_value(dataset=d, model=m, component=c, qualifier="us")   #xposition
                        vs = b.get_value(dataset=d, model=m, component=c, qualifier="vs")   #yposition
                        ws = b.get_value(dataset=d, model=m, component=c, qualifier="ws")   #zposition                        
                        theta_arcsec = np.arctan(us/distance)*(180/np.pi)*3600     #displacement in x-direction - arctan returns radians
                        phi_arcsec   = np.arctan(vs/distance)*(180/np.pi)*3600     #displacement in y-direction - arctan returns radians
                        alpha_arcsec = np.arctan(ws/distance)*(180/np.pi)*3600     #displacement in z-direction - arctan returns radians
                        binary[d][m][c]["theta"] = theta_arcsec
                        binary[d][m][c]["phi"] = phi_arcsec
                        binary[d][m][c]["alpha"] = alpha_arcsec
                        
                        for q in b[d][m][c].qualifiers:
                            binary[d][m][c][q] = b.get_value(dataset=d, model=m, component=c, qualifier=q) 
#                            print("dataset: %s, model: %s, component: %s, qualifier: %s"%(d,m,c,q))
            
            #extract all available lightcurve datasets of the created models
            if d[0:2] == "lc":
                binary[d][m]["phases"] = b.to_phase(b.get_value(dataset=d, model=m, qualifier="times"))
                for q in b[d][m].qualifiers:
                    binary[d][m][q] = b.get_value(dataset=d, model=m, qualifier=q) 
#                    print("dataset: %s, model: %s, qualifier: %s"%(d,m,q))
            
            if d[0:2] == "rv":
                for c in b[d][m].components:
                    binary[d][m][c] = {}
                    binary[d][m][c]["phases"] = b.to_phase(b.get_value(dataset=d, model=m, component=c, qualifier="times"))
#                    print("dataset: %s, model: %s, component: %s"%(d,m,c))

                    for q in b[d][m][c].qualifiers:
                        binary[d][m][c][q] = b.get_value(dataset=d, model=m, component=c, qualifier=q) 
#                        print("dataset: %s, model: %s, qualifier: %s"%(d,m,q))

                    
#    print(binary)
    
    #shows all available twigs, if set to true
    if show_keys:
        print("ALLOWED Keys/twigs")
        for k1 in binary.keys():
            print("\n_______twig %s__________\n" % (k1))
            for k2 in binary[k1].keys():
                if type(binary[k1][k2]) == bool:
                    print("binary[\"%s\"][\"%s\"]"%(k1,k2))
                
                elif k2 == "binary":
                    for k3 in binary[k1][k2].keys():
                        print("binary[\"%s\"][\"%s\"][\"%s\"]"%(k1,k2,k3))              
               
                else:
                    for k3 in binary[k1][k2].keys():
                        if type(binary[k1][k2][k3]) == np.ndarray:                    
                            print("binary[\"%s\"][\"%s\"][\"%s\"]"%(k1,k2,k3))
                        else:
                            for k4 in binary[k1][k2][k3].keys():
                                print("binary[\"%s\"][\"%s\"][\"%s\"][\"%s\"]"%(k1,k2,k3,k4))

    #time execution
    if timeit:
        task.end_task()   
        
    return binary

#______________________________________________________________________________
#function plot extracted data (data in binary-object)
#TODO: Add a variable to an array of color-names. Those will be used in order to plot each run    
def plot_data_PHOEBE(binary, lc_phase_time_choice="phases", rv_phase_time_choice="phases", orb_solrad_arcsec_choice = "arcsec", save_plot=False, save_path="", plot_lc=True, plot_orb=True, plot_rv=True, timeit=False):
    """
    creates plots of the extracted values
    to not show a plot, set plot_ to False
    to save plots to a defined location: set save_plot to True and define a save_path (string)
        the filename will be generated automatically
        if save_path is not defined, the current working directory will be used
        if not the whole path is given, the path will start from the current working directory
        save_path has to have a / at the end, otherwise it will be part of the name

    Parameters
    ----------
    binary : dict
        Object extracted from PHOEBE with extract_data function.
    lc_phase_time_choice : str, optional
        Used to specify wether to use time or phase for lc plotting. The default is "phases".
    rv_phase_time_choice : TYPE, optional
        Used to specify wether to use time or phase for rv plotting. The default is "phases".
    orb_solrad_arcsec_choice : TYPE, optional
        Used to specify wether to use solar radii or arcsec for orbit plotting. The default is "arcsec".
    save_plot : bool, optional
        If True, saves the plot to the specified path. The default is False.
    save_path : str, optional
        Used to specify the savepath of the created plots. The default is "".
    plot_lc : bool, optional
        If True, creates a plot of the LC. The default is True.
    plot_orb : TYPE, optional
        If True, creates a plot of the Orbits. The default is True.
    plot_rv : TYPE, optional
        If True, creates a plot of the RV. The default is True.
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        
    Returns
    -------
    None.

    """
    

    import matplotlib.pyplot as plt
    import os
    import re
    from utility_astroLuSt import Time_stuff
    
    #time execution
    if timeit:
        task = Time_stuff("plot_ax")
        task.start_task()
    
    #check if safe_path has the needed structure
    if (not save_path[-1] == "/") or (save_path == ""):
        print("ERROR: save_path has to have an \"/\" at the end!")
        return
    
    if not os.path.isdir(save_path):   #create new saveingdirectory for plots, if it does not exist
        os.makedirs(save_path)


    print("Started creating Plots")
    
        
    for nrd, d in enumerate(binary): 
        if not plot_lc:  binary[d]["enabled"] = False
        if not plot_orb: binary[d]["enabled"] = False
        if not plot_rv:  binary[d]["enabled"] = False
        
        #return message, if no plots are crated
        if (not plot_lc) and (not plot_orb) and (not plot_rv):
            print("No plots created.\nAll values binary[\"dataset\"][\"enabled\"] == False for all datasets")
            return
        
        if binary[d]["enabled"]:
            for nrm, m in enumerate(binary[d]):
                if m != "enabled" and m != "binary":
                    
                    #plot lightcurves
                    if d[0:2] == "lc":
                        
                        passband = re.findall("(?<=_).+", d)[0] #get which passband filter was used
                        
                        #figure
                        fignamelc = "lc%s"%(1000+10*nrd+nrm)
                        figlc = plt.figure(num=fignamelc, figsize=(10,10))
                        figlc.suptitle("lightcurve \nModel: %s, Passband: %s"%(m,passband), fontsize=18)
                        plt.xticks(fontsize=16); plt.yticks(fontsize=16)
                        plt.ylabel(r"flux $ \left[ \frac{W}{m^2} \right]$", fontsize=16)
                        
                        print("\nplotting %s(%s) of binary[%s][%s] in figure %s" % ("flux", lc_phase_time_choice, d,m, fignamelc))

                        #plot                            
                        fluxes = binary[d][m]["fluxes"]
                        #check wether to plot over phases or times                        
                        if lc_phase_time_choice == "times":
                            times = binary[d][m]["times"]                            
                            plt.xlabel(r"time [d]", fontsize=16)
                            plt.plot(times, fluxes, "o", figure=figlc, label=m+"_"+passband)
                        elif lc_phase_time_choice == "phases":
                            phases = binary[d][m]["phases"]                            
                            plt.xlabel(r"phases", fontsize=16)
                            plt.plot(phases, fluxes, "o", figure=figlc, label=m+"_"+passband)
                        else:
                            print("Neither keyword is correct\ Keyword has to be either \"times\" or \"phases\". Default is \"phases\"!")
                            return

                        plt.legend()
                    
                    #save
                    if save_plot:
                        filename = "plot_"+d+"_"+m+".png"
                        path = save_path + filename
                        plt.savefig(path)
                        print("plots saved to %s" %(path))
                        
                    #plot orbits    
                    if d[0:3] == "orb":
                        
                        #figure
                        fignameorb = "orb%s"%(2000+nrm) #create figure here, to show orbits of both components in one graph
                        figorb = plt.figure(num=fignameorb, figsize=(10,10))
                        figorb.suptitle("orbits \nModel:%s" %(m), fontsize=18)
                        plt.xticks(fontsize=16); plt.yticks(fontsize=16)

                        for nrc, c in enumerate(binary[d][m]):                            
                            print("\nplotting %s(%s) of binary[%s][%s][%s] in figure %s" % ("vs", "us", d,m,c, fignameorb))
                            
                            #plot
                            if orb_solrad_arcsec_choice == "solrad":
                                vs = binary[d][m][c]["vs"]
                                us = binary[d][m][c]["us"]
                                plt.xlabel(r"us (xposition) $ \left[R_{\odot} \right]$", fontsize=16)
                                plt.ylabel(r"vs (yposition) $ \left[R_{\odot} \right]$", fontsize=16)                                
                                plt.plot(us, vs, figure = figorb, label=c)
                            elif orb_solrad_arcsec_choice == "arcsec":
                                theta = binary[d][m][c]["theta"]
                                phi = binary[d][m][c]["phi"]
                                plt.xlabel(r"theta $ \left[ '' \right]$", fontsize=16)
                                plt.ylabel(r"phi $ \left[ '' \right]$", fontsize=16)                                
                                plt.plot(theta, phi, figure = figorb, label=c)
                            else:
                                print("Neither keyword is correct\ Keyword has to be either \"solrad\" or \"arcsec\". Default is \"arcsec\"!")
                            
                            plt.legend()
                            

                        #save
                        if save_plot:
                            filename = "plot_"+d+"_"+m+".png"
                            path = save_path + filename
                            plt.savefig(path)
                            print("plots saved to %s" %(path))
                    
                    #plot radial velocities        
                    if d[0:2] == "rv":
                        
                        #figure
                        fignamerv = "rv%s"%(3000+nrm) #create figure here, to show radial velocities of both components in one graph
                        figrv = plt.figure(num=fignamerv, figsize=(10,10))
                        figrv.suptitle("radial velocities \nModel:%s" %(m), fontsize=18)
                        plt.xticks(fontsize=16); plt.yticks(fontsize=16)
                        plt.ylabel(r"Radial Velocity $\left[ \frac{km}{s} \right]$", fontsize=16)

                        for nrc, c in enumerate(binary[d][m]):
                            
                            #plot
                            rvs = binary[d][m][c]["rvs"]
                            if rv_phase_time_choice == "times":
                                print("\nplotting %s(%s) of binary[%s][%s][%s] in figure %s" % ("rvs", "time", d,m,c, fignamerv))
                                times = binary[d][m][c]["times"]
                                plt.xlabel(r"time [d]", fontsize=16)
                                plt.plot(times, rvs, marker="o", linestyle="", figure=figrv, label=c)
 
#!!!!!to do: phase does not work propely!!!!!
                            elif rv_phase_time_choice == "phases":
                                print("\nplotting %s(%s) of binary[%s][%s][%s] in figure %s" % ("rvs", "phase", d,m,c, fignamerv))
                                phases = binary[d][m][c]["phases"]
                                plt.xlabel(r"phases", fontsize=16)
                                plt.plot(phases, rvs, marker="o", linestyle="", figure=figrv, label=c)
                             
                            else:
                                print("Neither keyword is correct\ Keyword has to be either \"times\" or \"phases\". Default is \"phases\"!")
                                return
                            
                            plt.legend()

                        #save
                        if save_plot:
                            filename = "plot_"+d+"_"+m+".png"
                            path = save_path + filename
                            plt.savefig(path)
                            print("plots saved to %s" %(path))

    #time execution
    if timeit:
        task.end_task()   

    
    plt.show()
    return
