from kSZ_forecast_general_func import *

class HOD_ELG_HSC:
    def __init__(self,model="NB912"):
        if model == "NB816":
            #lgM_c = 11.75
            #lgM_min = 12.46
            #sigma_lgM = 0.06
            #alpha = 1.06
            #F_c_A = 0.13
            #F_c_B = 0.95
            #F_s = 0.98
            #f_fake = 0.172

            self.lgM_c = 12.04
            self.lgM_min = 12.61
            self.sigma_lgM = 0.4
            self.alpha = 1.03
            self.F_c_A = 0.26
            self.F_c_B = 0.37
            self.F_s = 0.54
            self.f_fake = 0.14
        if model == "NB912":
            #lgM_c = 11.93
            #lgM_min = 12.47
            #sigma_lgM = 0.13
            #alpha = 1.23
            #F_c_A = 0.14
            #F_c_B = 0.9
            #F_s = 0.73
            #f_fake = 0.128    
            #'''
            self.lgM_c = 11.91
            self.lgM_min = 12.57
            self.sigma_lgM = 0.17
            self.alpha = 1.12
            self.F_c_A = 0.26
            self.F_c_B = 0.53
            self.F_s = 0.55
            self.f_fake = 0.104
        
    def number_centrals_mean(self,lgM):
        """
        Average number of central galaxies in each halo 
        Args:
            lgM:  array of the log10 of halo mass (Msun/h)
        Returns:
            array of mean number of central galaxies
        """
        term_gaussian = self.F_c_B*(1-self.F_c_A)*np.exp(-(lgM-self.lgM_c)**2/2/self.sigma_lgM**2)
        term_step = self.F_c_A * (1+special.erf((lgM-self.lgM_c)/self.sigma_lgM))
        return term_gaussian+term_step
        
    def number_satellites_mean(self,lgM):
        """
        Average number of satellites in each halo 
        Args:
            lgM:  array of the log10 of halo mass (Msun/h)
        Returns:
            array of mean number of satellites
        """
        delta_lgM = 1
        N_sat =  self.F_s * (1+special.erf((lgM-self.lgM_min)/delta_lgM))*(10**(lgM-self.lgM_min))**self.alpha
        return N_sat
    def get_number_centrals(self, lgM):
        """
        Randomly draw the number of satellite galaxies in each halo,
        brighter than mag_faint, from a Poisson distribution
        Args:
            lgM: array of the log10 of halo mass (Msun/h)
        Returns:
            array of number of satellite galaxies
        """
        number_mean = self.number_centrals_mean(lgM)
        # draw random number from binomial distribution
        return np.random.default_rng().binomial(1, number_mean)
    def get_number_satellites(self, lgM):
        """
        Randomly draw the number of satellite galaxies in each halo,
        brighter than mag_faint, from a Poisson distribution
        Args:
            lgM: array of the log10 of halo mass (Msun/h)
        Returns:
            array of number of satellite galaxies
        """
        number_mean = self.number_satellites_mean(lgM)
        # draw random number from Poisson distribution
        return np.random.poisson(number_mean)
class HOD_ELG_DESI:
    def __init__(self,model="L0"):
        if model == "L0":
            self.lgM_c = 11.234
            self.sigma_lgM = 0.206
            self.F_c_A = 0.133
            self.F_c_B = 0.01
            self.beta_c =-0.185
            self.lgM_min = 11.69
            self.F_s = 0.015
            self.delta_lgM= 0.516
            self.alpha = 0.947
        if model == "L1":
            self.lgM_c = 11.415
            self.sigma_lgM = 0.224
            self.F_c_A = 0.091
            self.F_c_B =0.146
            self.beta_c =-0.187
            self.lgM_min = 11.668
            self.F_s = 0.012
            self.delta_lgM= 0.516
            self.alpha = 0.939
        if model == "L2":
            self.lgM_c =11.528
            self.sigma_lgM = 0.241
            self.F_c_A = 0.035
            self.F_c_B = 0.075
            self.beta_c =-0.168
            self.lgM_min = 11.723
            self.F_s = 0.005
            self.delta_lgM= 0.508
            self.alpha = 0.94
        if model == "L3":
            self.lgM_c = 11.558
            self.sigma_lgM = 0.217
            self.F_c_A = 0.01
            self.F_c_B = 0.021
            self.beta_c = -0.065
            self.lgM_min = 11.783
            self.F_s = 0.001
            self.delta_lgM= 0.492
            self.alpha = 0.95
        
        
    def number_centrals_mean(self,lgM):
        """
        Average number of central galaxies in each halo 
        Args:
            lgM:  array of the log10 of halo mass (Msun/h)
        Returns:
            array of mean number of central galaxies
        """
        term_gaussian = self.F_c_B*(1-self.F_c_A)*np.exp(-(lgM-self.lgM_c)**2/2/self.sigma_lgM**2)
        term_step = self.F_c_A * (1+special.erf((lgM-self.lgM_c)/self.sigma_lgM))*(1+10**(lgM-self.lgM_c))**self.beta_c
        return term_gaussian+term_step
        
    def number_satellites_mean(self,lgM):
        """
        Average number of satellites in each halo 
        Args:
            lgM:  array of the log10 of halo mass (Msun/h)
        Returns:
            array of mean number of satellites
        """
        N_sat =  self.F_s * (1+special.erf((lgM-self.lgM_min)/self.delta_lgM))*(10**(lgM-self.lgM_min))**self.alpha
        return N_sat
    def get_number_centrals(self, lgM):
        """
        Randomly draw the number of satellite galaxies in each halo,
        brighter than mag_faint, from a Poisson distribution
        Args:
            lgM: array of the log10 of halo mass (Msun/h)
        Returns:
            array of number of satellite galaxies
        """
        number_mean = self.number_centrals_mean(lgM)
        # draw random number from binomial distribution
        return np.random.default_rng().binomial(1, number_mean)
    def get_number_satellites(self, lgM):
        """
        Randomly draw the number of satellite galaxies in each halo,
        brighter than mag_faint, from a Poisson distribution
        Args:
            lgM: array of the log10 of halo mass (Msun/h)
        Returns:
            array of number of satellite galaxies
        """
        number_mean = self.number_satellites_mean(lgM)
        # draw random number from Poisson distribution
        return np.random.poisson(number_mean)
class GalaxyCatalogueSnapshot():
    """
    Galaxy catalogue for a simuation snapshot
    Args:
        haloes:    halo catalogue
        cosmology: object of the class Cosmology
        box_size:  comoving simulation box length (Mpc/h)
    """
    def __init__(self, lgM_h,  cosmo = "WMAP7"):
        self._quantities = {}
        #self.redshift = redshift
        self.lgM_h = lgM_h
        #self.halox = halox
        self.cosmo = cosmo
        self.size = 0
        self.N_cen = 0
        self.N_sat = 0
        self.num_cen = 0
        self.num_sat = 0
        cosmology.setCosmology(cosmo)
    def get(self, prop):
        """
        Get property from catalogue

        Args:
            prop: string of the name of the property
        Returns:
            array of property
        """
        return self._quantities[prop]
    def add(self, prop, value):
        """
        Add property to catalogue

        Args:
            prop:  string of the name of the property
            value: array of values of property
        """
        self._quantities[prop] = value
    def add_galaxies(self, hod):
        """
        Use hod to randomly generate galaxy absolute magnitudes.
        Adds absolute magnitudes, central index, halo index,
        and central/satellite flag to the catalogue.
        Args:
            hod: object of the class HOD
        """
        # galaxy number
        self.num_cen = hod.get_number_centrals(self.lgM_h)
        self.num_sat = hod.get_number_satellites(self.lgM_h)
        self.halo_richness = self.num_cen+self.num_sat
        self.N_cen = np.sum(self.num_cen)
        self.N_sat = np.sum(self.num_sat)
        
        # update size of catalogue
        self.size = self.N_cen + self.N_sat
        
        # add boolean array of is central galaxy
        is_cen = np.zeros(self.size, dtype="bool")
        is_cen[:self.N_cen] = True
        self.add("is_cen", is_cen)
        
        # add index of host halo in halo catalogue
        id_ = np.arange(len(self.lgM_h))
        halo_ind_cen = np.where(self.num_cen>0)[0]
        halo_ind_sat = np.repeat(id_, self.num_sat)
        halo_ind = np.concatenate([halo_ind_cen, halo_ind_sat])
        self.add("halo_ind", halo_ind)
        
        #add halo mass
        self.add("halo_mass", self.lgM_h[halo_ind])
