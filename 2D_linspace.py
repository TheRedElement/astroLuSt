

#%%
# Creation of x1 and x2 component works
#TODO: combine the two axes!
def linspace_def_2D(x1_range, x2_range, means, covariances,
                 nintervals_x1=50, nintervals_x2=50, nbins=100, maxiter=100000,
                 go_exact=True, testplot=False, verbose=False, timeit=False):

    
    #import stuff
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as sps
    from numpy.linalg import det as det #determinante
    from numpy.linalg import inv as inv #inverse


    #time execution
    if timeit:
        time_stuff("linspace_def", head=True)

    
    
    #desired output field
    x1 = np.linspace(np.min(x1_range), np.max(x1_range), nintervals_x1+1)
    x2 = np.linspace(np.min(x2_range), np.max(x2_range), nintervals_x2+1)

    #create 2D scatter-field for distribution calculation and visualization
    x1_mg, x2_mg = np.meshgrid(x1, x2)
    X = np.dstack((x1_mg, x2_mg))

    #generate datapoint distribution
    #-------------------------------
    dist = 0
    for mu, Sigma in zip(means, covariances):
        gauss = sps.multivariate_normal.pdf(x=X, mean=mu, cov=Sigma)
        gauss /= gauss.max()        #to give all gaussians the same weight
        dist += gauss               #superposition all gaussians to get distribution

    dist /= np.sum(dist)            #scale so that it adds up to 1      
    dist*=nbins                     #scale to add up to the desired bins
    dist = np.sqrt(dist)            #because 2D and we want in total to be nbins bins

    #plot
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(x1_mg, x2_mg, dist, cmap="viridis")
    # ax.contourf(x1, x2, dist)
    plt.contourf(x1_mg, x2_mg, dist)
    # ax.elev = 90
    # ax.azimuth = 90
    plt.show()

    #generate linspace in x1-direction
    #---------------------------------

    fig2 = plt.figure()
    idx = 0             #for plotting
    totlen_x1 = 0       #to retrieve the total number of x1-points
    #get points for x1-direction
    x1_linspace = []
    for x12, d_row in zip(X, dist):
        x1_ls = np.array([])    #initiate array for x1 linspaces
        for x1, x1s, bins in zip(x12[:-1,0], x12[1:,0], d_row):
            # print(x1, x1s)
            int_bins = int(np.ceil(bins))       #convert number of bins to integer
            ls = np.linspace(x1, x1s, int_bins, endpoint=False) #create linspace for interval
            x1_ls = np.append(x1_ls, ls)        #add to whole x1-linspace

        #add border points
        # x1_ls = np.insert(x1_ls, 0, np.min(x1_range))
        # x1_ls = np.append(x1_ls, np.max(x1_range))

        #sort linspace
        x1_ls = np.sort(x1_ls)
        #delete all points outside of requested x1-range
        x1_ls = x1_ls[(np.min(x1_range)<=x1_ls)&(x1_ls<=np.max(x1_range))]
        
        #update total number of points
        totlen_x1 += len(x1_ls)

        #append to final x1 linspace
        x1_linspace.append(x1_ls)
        plt.plot(x1_ls, np.ones_like(x1_ls)*idx, "b.", alpha=0.5)
        idx += 1
    plt.show()
    
    print(f"Number of datapoints used on x1: {totlen_x1:d}")


    #generate linspace in x2-direction
    #---------------------------------
    
    fig3 = plt.figure()
    idx = 0             #for plotting
    totlen_x2 = 0       #to retrieve the total number of x2-points
    #get points for x2-direction
    x2_linspace = []
    X_swapped = np.swapaxes(X, 0,1)

    for x21, d_col in zip(X_swapped, dist.T):
        x2_ls = np.array([])    #initiate array for x2 linspaces
        for x2, x2s, bins in zip(x21[:-1], x21[1:], d_col):
            int_bins = int(np.ceil(bins))       #convert number of bins to integer
            ls = np.linspace(x2, x2s, int_bins, endpoint=False) #create linspace for interval
            x2_ls = np.append(x2_ls, ls)        #add to whole x1-linspace
        
        #add border points
        # x2_ls = np.insert(x2_ls, 0, np.min(x2_range))
        # x2_ls = np.append(x2_ls, np.max(x2_range))

        #sort linspace
        x2_ls = np.sort(x2_ls)
        #delete all points outside of requested x2-range
        x2_ls = x2_ls[(np.min(x2_range)<=x2_ls)&(x2_ls<=np.max(x2_range))]

        #update total number of points
        totlen_x2 += len(x2_ls)
        
        #append to final x2 linspace
        x2_linspace.append(x2_ls)
        plt.plot(np.ones_like(x2_ls)*idx, x2_ls, "r.", alpha=0.5)
        idx += 1
    plt.show()

    print(f"Number of datapoints used on x2: {totlen_x2:d}")
    print(f"total datapoints: {totlen_x1+totlen_x2:d}")
    print(f"requested datapoints: {nbins:d}")

    #combine linspaces of x1-, and x2-direction
    #------------------------------------------
    #Not working properly yet

    figls = plt.figure()
    idx = 0
    for x1_ls_, x2_ls_ in zip(x1_linspace, x2_linspace):
        # print(x1_ls_.shape, x2_ls_.shape)
        plt.plot(x1_ls_, np.ones_like(x1_ls_)*idx, "b.", alpha=0.5)
        plt.plot(np.ones_like(x2_ls_)*idx, x2_ls_, "r.", alpha=0.5)
        # maxpoints = np.min([x1_ls_.shape[0], x2_ls_.shape[0]])
        # print(maxpoints)
        # plt.plot(x1_ls_[:maxpoints], x2_ls_[:maxpoints], "b.", alpha=0.5)
        idx += 1
    plt.show()



    
    return #combined_linspace
    

##################################
# TEST FOR linspace_def function #  
##################################

import numpy as np
means = np.array([
    np.array([-2.5,-2.3]),
    np.array([3,3])
])
covariances = np.array([
    np.array([
        np.array([2,0]), np.array([0,1])
    ]),
    np.array([
        np.array([1,0]), np.array([0,5])
    ])    
])
x1_range=[-10,10]
x2_range=[-5,5]
nintervals_x1 = 20
nintervals_x2 = 20
nbins = 100
ls = linspace_def_2D(x1_range, x2_range, means=means, covariances=covariances,
                     nintervals_x1=nintervals_x1, nintervals_x2=nintervals_x2, nbins=nbins,
                     go_exact=True, testplot=True, verbose=True, timeit=False)

#%%

