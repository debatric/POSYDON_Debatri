
'''
This class contains all functions and parameters for pulsar evolution, e.g. NS spin, B-field, etc.
'''


__authors__ = [
    "Camille Liotine <cliotine@u.northwestern.edu>",
    "Abhishek Chattaraj <a.chattaraj@ufl.edu>",
    "Jorie McDermott <mcdermott.286@buckeyemail.osu.edu>"
]

import numpy as np
from posydon.utils import constants as const
from posydon.utils.common_functions import CO_radius
from astropy import constants as astro_const

"""
References
----------

[1] Chattopadhyay, D., Stevenson, S., Hurley, J. R., Rossi,
L. J., & Flynn, C. 2020, MNRAS, 494, 1587,
doi: 10.1093/mnras/staa756

[2] Ye, C. S., Kremer, K., Chatterjee, S., Rodriguez, C. L., &
Rasio, F. A. 2019, ApJ, 877, 122,
doi: 10.3847/1538-4357/ab1b21

[3] Zhang, C. M., & Kojima, Y. 2006, MNRAS, 366, 137,
doi: 10.1111/j.1365-2966.2005.09802.x

[4] Kiel, P. D., Hurley, J. R., Bailes, M., & Murray, J. R. 2008,
MNRAS, 388, 393, doi: 10.1111/j.1365-2966.2008.13402.x

[5] Lorimer, D. and Kramer, M. (2005). Handbook of Pulsar
Astronomy. Cambridge University Press.

[6] Li, Z., Chen, X., Chen, H.-L., and Han, Z. (2021). The
maximum accreted mass of recycled pulsars. The
Astrophysical Journal, 922(2):158.

[7] Haensel, P., Zdunik, J. L., Bejger, M., and Lattimer, J. M.
(2009). Keplerian frequency of uniformly rotating
neutron stars and strange stars. AA, 502(2):605–610.

[8] Shapiro, S. and Teukolsky, S. (1983). Black Holes, White
Dwarfs, and Neutron Stars: The Physics of Compact
Objects. John Wiley and Sons, Inc.

"""

class Pulsar:

    def __init__(self, initial_mass, B_field=float("NaN"), spin=float("NaN"), P_dot=float("NaN"), alpha = 30.0):
        '''
        Construct a Pulsar object.

        Parameters
        ----------
        float
            initial_mass: initial mass of the NS [Msun]
            B_field = "NaN": initial magnetic field [T]
            spin = "NaN": initial spin angular frequency [1/s]
            P_dot = "NaN": initial spindown rate [unitless]
            alpha = 30.0: rough value for the angle between the magnetic and rotational axis in degrees
        '''

        NS_RADIUS = CO_radius(initial_mass, "NS")*astro_const.R_sun.value  ## POSYDON constant for NS radius [m] 
        
        self.mass = initial_mass*astro_const.M_sun.value            ## mass of the NS [kg]
        self.radius = NS_RADIUS                               ## radius of the NS [m]
        self.moment_inertia = self.calc_moment_of_inertia()   ## moment of inertia of the NS  [kg m^2]
        ## Note: Because the NS radius is constant, moment of inertia should also be constant throughout NS evolution
        
        self.alpha = alpha*const.a2rad                        ## angle between magnetic and rotational axis [rad]
        self.Mdot_edd = self.calc_NS_edd_lim(np.nan)          ## Eddington accretion rate for the NS [Msun/yr]
        self.luminosity = self.calc_NS_luminosity()           ## radio luminosity of the pulsar [J/s]

        ## if magnetic field and spin not specified when initializing the pulsar,
        ## draw random values
        if np.isnan(B_field):
            self.Bfield = self.draw_NS_Bfield()      ## NS magnetic field [T]
        else:
            self.Bfield = B_field

        if np.isnan(spin):
            self.spin = self.draw_NS_spin()          ## NS spin angular frequency [1/s]
        else:
            self.spin = spin
            
        if np.isnan(P_dot):
            self.P_dot = self.calc_NS_spindown_rate()    ## initial calculation for NS spindown rate [unitless]
        else:
            self.P_dot = P_dot

        self.alive_state = self.is_alive()
 
    def draw_NS_spin(self):
        '''
        Draw the initial NS spin angular frequency Omega [1/s] from a uniform random distribution.
        Range is 0.1-2.0 sec for pulse period P.
        Omega = 2*pi/P
        '''
        return np.random.uniform(2*np.pi/.1, 2*np.pi/2) 

    def draw_NS_Bfield(self):
        '''
        Draw the initial NS B-field [T] from a lognormal distribution.
        Range is 10^11.5 - 10^13.8 Gauss.
        Value in Gauss is converted and returned in Tesla.
        '''
        # return np.random.uniform(3.16e11, 6.31e13)
        return (10**np.random.uniform(11.5, 13.8))*1e-4

    def calc_moment_of_inertia(self):
        '''
        Calculate the moment of intertia of the NS [kg*m^2]
        Eq. from Kiel et al. 2008 is in solar units
        '''
        M = self.mass/astro_const.M_sun.value       # mass of the NS [Msun] 
        R = self.radius/astro_const.R_sun.value     # radius of the NS [Rsun]

        return 2/7 * (1 - 2.42e-6*M/R - 2.9e-12*M**2/R**2)**-1 * M*R**2 * (astro_const.M_sun.value*astro_const.R_sun.value**2)
    
    def calc_NS_edd_lim(self, surface_h1):
        """
        Calculate the Eddington accretion rate for a NS [Msun/yr]
        Adapted from eddington_limit() in utils.common_functions
        Based on Mdot_E calculation in Chattopadhyay et. al. 2020

        Parameters
        ----------
        float
            surface_h1: surface hydrogen fraction of donor star
            If NaN, assume = 0.7155 (approximate surface H1 abundance
            for HMS stars @ Zsun)

        Returns
        -------
        float
            Mdot_edd: the Eddington accretion limit for the pulsar [Msun/yr]
        """
        if np.isnan(surface_h1): surface_h1 = 0.7155

        #eta = const.standard_cgrav*self.mass / (self.radius * const.clight**2)
        #Mdot_edd = (4*np.pi*const.standard_cgrav*self.mass) / (0.2*(1 + surface_h1)*eta*const.clight)  
        Mdot_edd = (4 * np.pi * astro_const.m_p.value * self.radius * astro_const.c.value) / astro_const.sigma_T.value
        return Mdot_edd / (astro_const.M_sun.value/const.secyer)      # convert [kg/s] to [Msun/yr]

    def calc_NS_luminosity(self):
        """
        Calculate the radio luminosity of the NS at 1400 MHz.
        units of distribution from Szary et al. 2014 are mJy*kpc^2
        Returns the radio luminosity in unis of [J/s]
        """
        kpc2m = astro_const.kpc.value    # kpc to m
        mJy2si = 1e-29              # milliJansky in SI units [J/s/m^2/Hz]
        Hz = 1400*1e6               # Hz conversion for radio luminosity (1400 MHz)
        mJykpc2si = mJy2si*kpc2m**2*Hz

        lum_draw = np.random.lognormal(0.5, 1.0)
        luminosity = 10**lum_draw*mJykpc2si
        return luminosity

    def calc_NS_spindown_rate(self):
        """
        Calculate the spindown rate of the NS.
        From Chattopadhyay et al. 2020
        """
        I = self.moment_inertia # get the moment of interia [kg m^2]
        alpha = self.alpha # angle between magnetic and rotational axis [rad]

        # get the change in spin [1/s^2], from Chattopadhyay et al. 2020
        omega_dot = - (8 * np.pi * (self.Bfield**2) * (self.radius**6) * (self.spin**3) * (np.sin(alpha))**2) / (3 * astro_const.mu0.value * (astro_const.c.value**3) * I)
        
        #P_dot =  9.87e-48*const.secyer * self.Bfield**2/P    # NS spindown rate
        P = (2*np.pi)/self.spin      # NS spin period
        P_dot = - (omega_dot * P) / self.spin # NS spindown rate from Chattopadhyay et al. 2020
        
        return P_dot
    
    def calc_NS_spindown_power(self):
        """
        Calculate the spindown power [J/s] of the NS.
        From Szary et al. 2019
        """
        P = (2*np.pi)/self.spin      # NS spin period
        P_dot = self.P_dot         # NS spindown rate

        I = self.moment_inertia # get the moment of interia [kg m^2]
        
        power = 4*np.pi**2 * I * P_dot / P**3 # spindown power [J/s]
        return power 

    def calc_magnetosphere_radius(self, Mdot_acc):
        '''
        Calculate the magnetospheric radius of the NS.
        From Chattopadhyay et. al. 2020

        Parameters
        ----------
        float
            Mdot_acc: accretion rate onto the NS [Msun/yr]

        Returns
        -------
        float
            r_m: alfven radius of the NS [m]
        '''
        Mdot_acc *= (astro_const.M_sun.value/const.secyer)  # convert [Msun/yr] to [kg/s]

        #eta = 0.4    # constant depending on disk-magnetosphere interaction
        #mu = self.Bfield * self.radius**3     # NS mangetic moment
        #r_m = eta * (mu**4/(2*const.standard_cgrav*self.mass*Mdot_acc**2))**(1/7)

        # get the alfven radius [m], from Chattopadhyay et. al. 2020
        r_m = ((2 * np.pi**2 * self.radius**12 * self.Bfield**4)/(astro_const.G.value * astro_const.mu0.value**2 * Mdot_acc**2 * self.mass))**(1/7)
        
        return r_m

    def calc_NS_spin_equilibrium(self, Mdot_acc):
        '''
        Calculate the equilibrium spin of the NS.

        Parameters
        ----------
        float
            Mdot_acc: accretion rate onto the NS [Msun/yr]

        Returns
        -------
        float
            omega_eq: equillibrium spin of the NS [1/s]
        '''
        r_m = self.calc_magnetosphere_radius(Mdot_acc) # Alfven radius [m]
        
        omega_eq = 1/(2*np.pi) * (astro_const.G.value*self.mass/r_m**3)**(1/2) # equillibrium spin [1/s]
        
        return omega_eq

    def calc_NS_spin_limit(self, C = 1.15):
        '''
        Calculate the Keplarian limit for the NS spin frequency.
        This limit is discussed in Li et. al. 2021.

        Parameters
        ----------
        float
            C = 1.15: a fitted parameter from  Haensel et al. 2009

        Returns
        -------
        float
            neutron star spin frequency at the Keplarian limit [1/s]
        '''
        R_ns = self.radius/(1e3*10) # convert the neutron star radius from m to units of 10 km
        f_k = C * np.sqrt(self.mass/astro_const.M_sun.value) * (R_ns**(-3/2)) # unints of kHz
        return f_k * 1e3 # return the spin frequency in units of Hz for more convenient use

    def Vdiff_fnct(self, R_mag):
        '''
        Calculate the difference between the angular velocity of the
        neutron star at the magnetic radius (Omega_k) and the co-rotation
        angular velocity (Omega_co).
        From Chattopadhyay et al. 2020.

        Parameters
        ----------
        float
            R_mag: magnetic radius of the pulsar; half the Alfven radius [m]

        Returns
        -------
        float
            the difference between the angular velocity at the magnetic radius and the co-rotation radius [1/s]
        '''

        Omega_co = (2*np.pi)*self.spin # get the angular velocity of rotation (same as the co-rotation angular velocity) [Hz]  
        
        Omega_k = np.sqrt(astro_const.G.value*self.mass/R_mag**3) # from COMPAS function [Hz]
        
        return Omega_k - Omega_co # units of Hz
   
    def detached_evolve(self, delta_t, tau_d):
        '''
        Evolve a pulsar during detached evolution.

        Parameters
        ----------
        float
            delta_t: the duration of the detached evolution phase [yr]
            tau_d: the Bfield decay timescale [Gyr]
        '''
        alpha = self.alpha              # angle between the axis of rotation and magnetic axis [rad]
        mu_0 = astro_const.mu0.value          # permeability of free space [N/A^2]
        B_min = 1e8*1e-4                # minimum Bfield strength at which Bfield decay ceases [T]
        tau_d *= (1e9*const.secyer)     # B-field decay timescale [s]

        delta_t *= const.secyer         # convert detached evolution time [s]

        I = self.moment_inertia         # moment of interia of the NS [kg m^2]
        R = self.radius                 # radius of the NS [m]
        B_i = self.Bfield               # magnetic field of the NS [T]

        # evolve the NS B-field
        B_f = (B_i - B_min) * np.exp(-delta_t/tau_d) + B_min    # [T]
        self.Bfield = B_f

        # evolve the NS spin
        A = 8 * np.pi * R**6 * np.sin(alpha)**2 / (3 * mu_0 * astro_const.c.value**3 * I)    # [s^5 A^2/kg^2]
        omega_f = np.sqrt(1/( A*(B_min**2*delta_t - tau_d*B_min*(B_f - B_i) - tau_d/2*(B_f**2 - B_i**2))  + 1/self.spin**2))    # [1/s]
        self.spin = omega_f

        # get the new NS spindown rate
        P_dot = self.calc_NS_spindown_rate()
        self.P_dot = P_dot

        # check if pulsar crossed the death line
        self.alive_state = self.is_alive()

    
    def RLO_evolve_CMC(self, delta_t, tau_d, delta_M, delta_Md):
        '''
        Evolve a pulsar during Roche Lobe overflow (RLO).

        Parameters
        ----------
        float
            delta_t: the duration of the RLO accretion phase [yr]
            tau_d: the Bfield decay timescale [Gyr] 
            delta_M: the total amount of mass accreted by the pulsar during RLO [Msun]
            delta_Md: the magmetic field mass decay scale [Msun]
        '''

        G = astro_const.G.value             # gravitational constant [c^3 kg^-1 s^-2]
        mu_0 = astro_const.mu0.value        # permeability of free space [N/A^2]
        tau_d *= (1e9*const.secyer)   # B-field decay timescale [s]
        delta_Md *= astro_const.M_sun.value # magnetic field mass decay scale [kg]

        delta_M *= astro_const.M_sun.value  # mass accreted by pulsar during RLO [kg]
        delta_t *= const.secyer       # duration of RLO accretion [s]

        M_i = self.mass               # mass of the NS before accretion [kg]
        B_i = self.Bfield             # B-field of the NS before accretion [T]
        R = self.radius               # radius of the NS [m]
        I = self.moment_inertia       # Moment of inertia [kg m^2]
        K_const = 3.1126032e-40*1e8   # convert K in G^-2 to T^-2 [s/T^2]


        # evolve the NS spin
        J_i = 2/5*M_i*R**2*self.spin    # spin angular momentum (J) of the NS before accretion [kg m^2/s]
        #J_i = I*self.spin
        M_f = M_i + delta_M
        
        # spin-up from accretion
        omega_k = np.sqrt(G*M_i/R**3)    # [1/s]
        delta_J_acc = 2/5*delta_M*R**2*omega_k    # [kg m^2/s]

        # spin down from dipole radiation
        delta_omega = K_const*B_i**2*self.spin**3    # [1/s^2]
        delta_J_rad = 2/5*M_f*R**2*delta_omega    # [kg m^2/s^2]
                
        J_f = J_i + delta_J_acc - (delta_J_rad*delta_t)    # CHECK THIS, DOES MULTIPLYING DELTA_J_RAD BY DELTA_T COMPLETLY FIX THE ISSUE???
        
        self.mass = M_f

        omega_f = J_f/(2/5*M_f*R**2)
        #omega_f = J_f/I
        self.spin = omega_f

        # evolve the NS B-field
        B_min = 5e7 * 1e-4    # convert from G to T [T]
        B_f = B_i/(1 + delta_M/delta_Md) * np.exp(-delta_t/tau_d) + B_min    # [T]
        self.Bfield = B_f

        # get the new spindown rate of the NS
        P_dot = self.calc_NS_spindown_rate()
        self.P_dot = P_dot

        # check if pulsar has crossed the death line
        self.alive_state = self.is_alive()
    
    def RLO_evolve_COMPAS(self, delta_M, delta_Md, Mdot_acc, CE):
        '''
        Evolve a pulsar during Roche Lobe overflow (RLO).
        This uses the prescription for B-field decay applied in Chattopadhyay et al. 2020 from Oslowski et al. 2011.
        Spin-down is the same for now.

        Parameters
        ----------
        delta_M: float, total mass accreted onto the pulsar during RLO [Msun]
        delta_Md: float, mass decay scale (free parameter) [Msun]
        Mdot_acc: float, mass accretion rate onto the pulsar  [Msun/yr]
        CE: bool, True/False if the pulsar is accreting in a common envelope state
        '''
        G = astro_const.G.value             # gravitational constant [m^3 kg^-1 s^-2]
        delta_Md *= astro_const.M_sun.value # magnetic field mass decay scale [kg]
        B_min = 1e8*1e-4              # minimum Bfield strength at which Bfield decay ceases [T]
        mu_0 = astro_const.mu0.value        # permeability of free space [N/A^2]
        efficiency = 1                # set the pulsar to the maximum accretion efficiency to start

        delta_M *= astro_const.M_sun.value  # convert Msun to kg

        M_i = self.mass               # mass of the NS before accretion [kg]  
        R = self.radius               # radius of the NS [m]
        B_i = self.Bfield             # B-field of the NS before accretion [T]
        I = self.moment_inertia       # moment of interia of the NS [kg m^2]

        R_mag = self.calc_magnetosphere_radius(Mdot_acc)   # calculate magnetic radius BEFORE B-field decay and mass accretion [m]
        
        # allow the NS to accrete mass
        M_f = M_i + (delta_M * efficiency) # add an accretion efficiency factor to prevent the pulsar from accreting when it is spinning too fast [kg]
        self.mass = M_f

        #R_alfven = (2*np.pi**2/(G*mu_0**2))**(1/7) * (R**6/(self.Mdot_edd*M_i**(1/2)))**(2/7) * B_i**(4/7) # Alfven radius
        #R_mag = R_alfven/2   # magnetic radius
   
        # evolve the NS B-field
        B_f = (B_i - B_min)*np.exp(-delta_M/delta_Md) + B_min    # [T]
        self.Bfield = B_f

        #R_mag = self.calc_magnetosphere_radius(Mdot_acc)   # calculate magnetic radius AFTER B-field decay and mass accretion
        
        # evolve the NS spin
        J_i = I*self.spin   # spin angular momentum (J) of the NS before accretion [kg m^2/s]
       
        # do we need to subtract the spin from the co-rotation radius here? probably?
        # last we discussed with Vicky, we decided to leave this out until track interpolation
        # add an ODE solver using the dM/dt we have in the meantime?
        #omega_k = np.sqrt(G*M_i/R_mag**3) 
        #delta_J = 2/5*delta_M*R_mag**2*omega_k  #(omega_k - omega_co)  # change in J due to accretion
        #J_f = J_i + delta_J

        # calculate the rate of change of angular momentum using Vdiff
        V_diff = np.mod(self.Vdiff_fnct(R_mag))    # [1/s]
        J_dot = efficiency * V_diff * (R_mag**2) * ((Mdot_acc*astro_const.M_sun.value)/const.secyer)    # [kg m^2/s^2]

        # calculate the amount of time in each delta_M step and use this to calculate delta_J (IS THIS THE RIGHT WAY TO FIX THE ISSUE OF J_DOT_ACC UNIT MISMATCH???)
        delta_t = delta_M/((Mdot_acc*astro_const.M_sun.value)/const.secyer)    # [s]
        delta_J = J_dot*delta_t    # [kg m^2/s]
        
        J_f = J_i + delta_J # [kg m^2/s]

        R_mag = self.calc_magnetosphere_radius(Mdot_acc)   # calculate magnetic radius AFTER B-field decay and mass accretion [m]
        
        # double check if this should be R_mag or R_NS
        # CHECK THESE UNITS!!
        I_mag = 0.237 * M_f * (R_mag**2) * (1 + (4.2 * (M_f/R_mag)*(1e3/astro_const.M_sun.value)) + 90*((M_f/R_mag)*(1e3/astro_const.M_sun.value))**4) # kg m^2 (from Chattopadhyay, et. al. 2020)
        omega_f = J_f/I_mag # [Hz]
        self.spin = omega_f

        # check if pulsar has reached the maximum spin limit 
        spin_limit = self.calc_NS_spin_limit() # units of Hz
        if self.spin >= spin_limit: efficiency = 0 # prevent accretion when the pulsar hits this spin limit
        # (does not apply for CE accretion)
        #if not CE:
        spin_eq = self.calc_NS_spin_equilibrium(Mdot_acc)
        if self.spin > spin_eq: self.spin = spin_eq

        # get the new spindown rate of the NS
        P_dot = self.calc_NS_spindown_rate()
        self.P_dot = P_dot

        # check if pulsar crossed the death line
        self.alive_state = self.is_alive()

    def CE_evolve(self, CE_acc_prescription, acc_decay_prescription, acc_lower_limit, 
                  M_comp, R_comp, delta_Md, delta_t, tau_d):
        '''
        Evolve a pulsar during common envelope, accounting for mass accretion onto the NS.
        '''   
        if CE_acc_prescription == "None": delta_M = 0
        
        elif CE_acc_prescription == "uniform":
            # assume amount of mass accreted during CE is 0.04-0.1 Msun  
            delta_M = np.random.uniform(0.04, 0.1)    # [Msun]

        elif CE_acc_prescription in ["MacLeod", "MacLeod_bounded"]:
            # use the MacLeod prescription from Chattopadhyay et al. 2020, fit to Fig. 4 in Macleod & Ramirez-Ruiz 
            a_a = -1.1e-5; a_b = 1.5e-2; b_a = 1.2e-4; b_b = -1.5e-1

            a = a_a*M_comp + b_a
            b = a_b*M_comp + b_b
            delta_M = np.abs(a*R_comp + b)

            if delta_M > 0.1: delta_M = 0.1

            # assume amount of mass accreted during CE is 0.04-0.1 Msun  
            if CE_acc_prescription == "MacLeod_bounded": 
                if delta_M < 0.04: delta_M = 0.04

        if acc_decay_prescription == "CMC":
            self.RLO_evolve_CMC(delta_t, tau_d, delta_M, delta_Md)
        elif acc_decay_prescription == "COMPAS":
            self.RLO_evolve_COMPAS(delta_M, delta_Md, self.Mdot_edd, True)

    def is_alive(self):
        '''
        Check if the pulsar has crossed the death line.
        ''' 
        P = 2*np.pi/self.spin     # spin period of the pulsar [s]
        P_dot = self.calc_NS_spindown_rate() # spindown rate of the pulsar

        death_line = 0.17e12*P**2   # death line from Ruderman & Sutherlandt 1975

        E_max = 0.01   # threshold radio efficiency
        L = self.luminosity # luminosity of the NS [J/s]
        Edot = self.calc_NS_spindown_power() # spindown power of the NS [J/s]
  
        if ((L/Edot) < E_max): return True
        #elif (self.Bfield < death_line): return False
        else: return False
        
