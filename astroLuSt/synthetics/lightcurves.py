

#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import re
from typing import Union, Tuple, Callable
import warnings


from astroLuSt.preprocessing.dataseries_manipulation import periodic_shift


#%%definitions
class SnythEB:

    def __init__(self,
        mu1:float, mu2:float, mu3:float,
        sigma1:float, sigma2:float, sigma3:float,
        freq3:float,
        scale1:float, scale2:float, scale3:float,
        dip1_add:float, dip2_add:float, noise_scale:float, total_shift:float,
        fluxmin:float,
        dip_border_factor:float=4.5, resolution:float=100,
        total_eclipse1:bool=False, total_eclipse2:bool=False,
        verbose:int=0
        ) -> None:

        self.mu1               = mu1
        self.mu2               = mu2
        self.mu3               = mu3
        self.sigma1            = sigma1
        self.sigma2            = sigma2
        self.sigma3            = sigma3
        self.freq3             = freq3
        self.scale1            = scale1
        self.scale2            = scale2
        self.scale3            = scale3
        self.dip1_add          = dip1_add
        self.dip2_add          = dip2_add
        self.noise_scale       = noise_scale
        self.total_shift       = total_shift
        self.fluxmin           = fluxmin
        self.resolution        = resolution
        self.dip_border_factor = dip_border_factor
        self.total_eclipse1    = total_eclipse1
        self.total_eclipse2    = total_eclipse2

        self.verbose = verbose
        
        return
    
    def __repr__(self) -> str:
        return (
            f'TODO: IMPLEMENT!!!!!!'
            f'SynthEB(\n'
            f'    ,\n'
            f')'
        )

    def gaussian(self,
        times, mu, sigma
        ):
        """
            - function to create a gaussian with center mu and stadard-deviation sigma
        """
        Q1 = sigma*np.sqrt(2*np.pi)
        exp1 = -0.5*(times-mu)**2 / sigma**2
        
        return np.exp(exp1)/Q1
    
    def generate(self):

        #check if input values are correct
        warningmessage1 = "%s should be in the interval [%g, %g] in order to avoid overlapping dips!"
        warningmessage2 = "%s should be in the interval [%g, %g]!"
        warningmessage4 = "%s has to be in the interval [%g, %g], in order to prevent empty masks!"
        if self.mu1 < -0.3 or -0.2 < self.mu1:
            print(warningmessage1%("self.mu1", -0.3, -0.2))
        if self.mu2 < 0.2 or 0.3 < self.mu2:
            print(warningmessage1%("self.mu2", 0.2, 0.3))
        if self.mu3 < -0.5 or 0.5 < self.mu3:
            print(warningmessage1%("self.mu3", -0.5, -0.5))
        if sigma1 < 0.01 or 0.045 < sigma1:
            print(warningmessage1%("sigma1", 0.01, 0.045))
        if sigma2 < 0.01 or 0.045 < sigma2:
            print(warningmessage1%("sigma2", 0.01, 0.045))
        if self.frequ3 < 0 or 6 < self.frequ3:
            print(warningmessage2%("self.frequ3", 0, 4))
        if self.scale1 < 0 or 0.04 < self.scale1:
            print(warningmessage1%("self.scale1", 0, 0.04))
        if self.scale2 < 0 or 0.04 < self.scale2:
            print(warningmessage1%("self.scale2", 0, 0.04))
        if self.scale3 < 0 or 0.15 < self.scale3:
            print(warningmessage2%("self.scale3", 0, 0.03))
        if self.dip1_add < 1e-3 or 0.01 < self.dip1_add:
            raise ValueError(warningmessage4%("self.dip1_add", 0, 0.01))
        if self.dip2_add < 1e-3 or 0.01 < self.dip2_add:
            raise ValueError(warningmessage4%("self.dip2_add", 0, 0.01))
        if self.total_shift < -0.5 or 0.5 < self.total_shift:
            raise ValueError("self.total_shift has to be in the interval [-0.5,0.5]")

        #set some initialization variables
        phases = np.linspace(-0.5, 0.5, self.resolution)

        #shift x_vals to enable dip reaching over boundary
        phases_shifted = periodic_shift(phases,               self.total_shift, [-0.5,0.5])
        mu1_shifted    = periodic_shift(np.array([self.mu1]), self.total_shift, [-0.5,0.5])[0]
        mu2_shifted    = periodic_shift(np.array([self.mu2]), self.total_shift, [-0.5,0.5])[0]
        
        #first dip
        if self.total_eclipse1:
            sigma1 = 0.5*np.abs((self.mu1+self.dip1_add) - (self.mu1-self.dip1_add))
        dip1_center = -self.scale1*self.gaussian(phases, self.mu1, sigma1)*(not self.total_eclipse1) #0 if total eclipse
        dip1_pos    = -self.scale1*self.gaussian(phases, self.mu1+self.dip1_add, sigma1)
        dip1_neg    = -self.scale1*self.gaussian(phases, self.mu1-self.dip1_add, sigma1)
        dip1        = dip1_neg + dip1_center + dip1_pos
            
        #second dip
        if self.total_eclipse2:
            sigma2 = 0.5*np.abs((self.mu2+self.dip2_add) - (self.mu2-self.dip2_add))
        dip2_center = -self.scale2*self.gaussian(phases, self.mu2, sigma2)*(not self.total_eclipse2) #0 if total eclipse
        dip2_pos    = -self.scale2*self.gaussian(phases, self.mu2+self.dip2_add, sigma2) 
        dip2_neg    = -self.scale2*self.gaussian(phases, self.mu2-self.dip2_add, sigma2)
        dip2        = dip2_neg + dip2_center + dip2_pos

        #basline flux
        # f_bl = self.scale3*gaussian(phases, self.mu3, sigma3)
        f_bl = self.scale3*np.sin(2*self.frequ3*np.pi*(phases + self.mu3))

        #put everything together to get synthetic LC
        fluxes = f_bl + dip1 + dip2
        noise = np.random.normal(size=np.shape(fluxes))*self.noise_scale
        relative_fluxes = np.interp(fluxes, (fluxes.min(), fluxes.max()), (self.fluxmin,1))  #normalize to 1 (such that sinusoidal baseline is centered around 1)
        relative_fluxes += noise   #add some noise

        #classify dip borders
        dip1_border1 = periodic_shift(np.array([self.mu1-np.abs(self.dip_border_factor*sigma1)]), self.total_shift, [-0.5,0.5])[0]
        dip1_border2 = periodic_shift(np.array([self.mu1+np.abs(self.dip_border_factor*sigma1)]), self.total_shift, [-0.5,0.5])[0]
        dip2_border1 = periodic_shift(np.array([self.mu2-np.abs(self.dip_border_factor*sigma2)]), self.total_shift, [-0.5,0.5])[0]
        dip2_border2 = periodic_shift(np.array([self.mu2+np.abs(self.dip_border_factor*sigma2)]), self.total_shift, [-0.5,0.5])[0]

        #masks to separate dips 
        if dip1_border1 < dip1_border2:
            mask1 = (dip1_border1<phases_shifted)&(phases_shifted<dip1_border2) #mask of dip1
        else:
            mask1 = (dip1_border1<phases_shifted)|(phases_shifted<dip1_border2) #mask of dip1 if reaching over border
        if dip2_border1 < dip2_border2:
            mask2 = (dip2_border1<phases_shifted)&(phases_shifted<dip2_border2) #mask of dip2
        else:
            mask2 = (dip2_border1<phases_shifted)|(phases_shifted<dip2_border2) #mask of dip2 if reaching over border


        #classify if total/partial eclipse
        std_crit = 2e-2
        d1_std = np.std(relative_fluxes[mask1])
        d2_std = np.std(relative_fluxes[mask2])
        
        if self.total_eclipse1:
            #total if specified
            d1_type = "total"
        elif d1_std <= std_crit:
            #non-eclipsing if indistinguishable from baseline
            d1_type = "non-eclipsing"
        else:
            #partial otherwise
            d1_type = "partial"

        if self.total_eclipse2:
            #total if specified
            d2_type = "total"
        elif d2_std <= std_crit:
            #non-eclipsing if indistinguishable from baseline
            d2_type = "non-eclipsing"
        else:
            #partial otherwise
            d2_type = "partial"
        
        #assign classification corretly
        if np.min(dip1) < np.min(dip2):
            d1_lab = f"Primary eclipse ({d1_type})"
            d2_lab = f"Secondary eclipse ({d2_type})"
        else:
            d1_lab = f"Secondary eclipse ({d1_type})"
            d2_lab = f"Primary eclipse ({d2_type})"    

        return {
            "fluxes":fluxes,
            "relative_fluxes":relative_fluxes,
            "phases":phases,
            "shifted_phases":phases_shifted,
            "dip_positions":[mu1_shifted, mu2_shifted],
            "dip_borders":[dip1_border1,dip1_border2,dip2_border1,dip2_border2],
            "dip1_type":d1_type,
            "dip2_type":d2_type,
            "dip1":dip1,
            "dip1_center":dip1_center,
            "dip1_pos":dip1_pos,
            "dip1_neg":dip1_neg,
            "dip2":dip2,
            "dip2_center":dip2_center,
            "dip2_pos":dip2_pos,
            "dip2_neg":dip2_neg,
            "basline_flux":f_bl,
            "noise":noise,
        }

    def plot_result(self):
            
        dip1_color = "tab:blue"
        dip2_color = "tab:orange"
        bl_color = "tab:grey"
        d1_color = "tab:blue"
        d2_color = "tab:orange"
        
        #insert nan to avoid connecting start and end
        insidx = np.argmin(phases_shifted)
        phases_shifted_plt = np.insert(phases_shifted, insidx, np.NaN)
        relative_fluxes_plt = np.insert(relative_fluxes, insidx, np.NaN)
        noise_plt = np.insert(noise, insidx, np.NaN)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_title("Generation parts", fontsize=16)
        ax1.plot(phases, dip1, color=dip1_color, alpha=1)#, label=d1_lab)
        ax1.plot(phases, dip1_center, color=dip1_color, alpha=.5)
        ax1.plot(phases, dip1_pos, color=dip1_color, alpha=.5)
        ax1.plot(phases, dip1_neg, color=dip1_color, alpha=.5)
        ax1.plot(phases, dip2, color=dip2_color, alpha=1)#, label=d2_lab)
        ax1.plot(phases, dip2_center, color=dip2_color, alpha=.5)
        ax1.plot(phases, dip2_pos, color=dip2_color, alpha=.5)
        ax1.plot(phases, dip2_neg, color=dip2_color, alpha=.5)
        ax1.plot(phases, f_bl, color=bl_color, alpha=.5, label="Baseline flux")
        # ax1.plot(phases, fluxes, color=whole_color, alpha=.7, label="Synthetic lightcurve")
        ax1.set_xticks([])
        ax1.set_ylabel("Flux-like", fontsize=16)
        ax2 = fig.add_subplot(212)
        ax2.set_title("Generated LC", fontsize=16)
        ax2.plot(phases_shifted[~mask1&~mask2], relative_fluxes[~mask1&~mask2], color="tab:green", marker=".", linestyle="", alpha=1, label="Continuum flux")
        ax2.plot(phases_shifted[mask1], relative_fluxes[mask1], color=d1_color, marker=".", linestyle="", alpha=1, label=d1_lab)
        ax2.plot(phases_shifted[mask2], relative_fluxes[mask2], color=d2_color, marker=".", linestyle="", alpha=1, label=d2_lab)
        ax2.plot(phases_shifted_plt, relative_fluxes_plt-noise_plt, color="k", alpha=1, label="Non-noisy LC")
        ax2.vlines((mu1_shifted, mu2_shifted), ymin=relative_fluxes.min(), ymax=relative_fluxes.max(), color="grey", linestyles="--")#, label="Dippositions")
        ax2.vlines((dip1_border1, dip1_border2, dip2_border1, dip2_border2), ymin=relative_fluxes.min(), ymax=relative_fluxes.max(), color="gainsboro", linestyles="--")#, label="Dipborders")
        ax2.set_xlabel("Phase", fontsize=16)
        ax2.set_ylabel("Relative-flux", fontsize=16)
        fig.legend()
        plt.tight_layout()
        plt.show()

        axs = fig.axes

        return fig

