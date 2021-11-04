
    ###################
    #Steinwender Lukas#
    ###################

#______________________________________________________________________________
#Class useful for working with PHOEBE
#TODO: update to extract more values
#TODO: finish documentation
class PHOEBE_additional:

    def __init__(self):
        pass

    def extract_data(phoebes_b, distance, unit_distance, show_keys=False, timeit=False):
        """
            Function to extract data from PHOEBE bundle

            Parameters
            ----------
            - phoebes_b
                - PHOEBE-bundle
                    - data structure object of PHOEBE to be converted to dict
            - distance
                - float
                - distance to observed object. If not known simply type 1
            - unit_distance
                - str
                - a string to specify the unit of distance
            - show_keys
                - bool, optional
                - If true prints out all possible keys (twigs) of the created dictionary
                - The default is False
            - timeit
                - bool, optional
                - Specify wether to time the task ad return the information or not
                - The default is False

            Returns
            -------
            - binary
                - dictionary of PHOEBEs default binary object with all the data-arrays in it.
                - The structure is the following:
                - binary = {
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
                - to add new dataset: add another if d[...:...] == "newdataset", and extract the date with phoebes twig-system
        """
        
        import numpy as np
        from module_parts.utility_astroLuSt import Time_stuff
        
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

