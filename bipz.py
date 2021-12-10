import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.special import erf, gamma
from classy import Class

print('[Initialising CLASS]\n')

c = 299792.458 #km/s
h = 0.674
ns = 0.965
s8 = 0.811
Om = 0.315
Ob = 0.049
Ok = 0.000
N_ur = 2.99
N_ncdm = 0.0
w = -1.0

print('(Pre-defined cosmology)')
print('      h:', h)
print('    n_s:', ns)
print('     s8:', s8)
print('   N_ur:', N_ur)
print(' N_ncdm:', N_ncdm)
print('Omega_m:', Om)
print('      w:', w )
print('\n')


# Initialise CLASS (linear power spectra)
class_settings = {'output': 'mPk', 
                  'lensing': 'no',
                  'h': h, 
                  'n_s': ns,
                  'sigma8': s8, 
                  'Omega_cdm': Om-Ob, 
                  'Omega_b': Ob,
                  'z_max_pk': 10.0,
                  'P_k_max_1/Mpc': 250}
pclass = Class()
pclass.set(class_settings)
pclass.compute()
        
# Non-linear power spectra (Halofit)
class_settings_nl = {'output': 'mPk', 
                  'lensing': 'no',
                  'h': h, 
                  'n_s': ns,
                  'sigma8': s8, 
                  'Omega_cdm': Om-Ob, 
                  'Omega_b': Ob,
                  'non linear': 'halofit',
                  'z_max_pk': 10.0,
                  'P_k_max_1/Mpc': 250}

pnlclass = Class()
pnlclass.set(class_settings_nl)
pnlclass.compute()

# Get background quantities
bg       = pclass.get_background()
H        = interp1d(bg['z'],(c/h)*bg['H [1/Mpc]'])
comov    = interp1d(bg['z'],h*bg['comov. dist.'])
redshift = interp1d(h*bg['comov. dist.'],bg['z'])
Dz       = interp1d(bg['z'],bg['gr.fac. D'])

print('[CLASS Initialised]')

SKA = {
    "name" : "SKA_SD",
    "dish_size" : 15.0,
    "t_inst" : 50.0,
    "t_total" : 10000.0,
    "n_dish" : 133,
    "area_eff" : 1.0,
    "im_type" : "single_dish",
    "base_file" : "none",
    "base_min" : 0.0,
    "base_max" : 15.0,
    "omega_sky" : 20000.0
}

HIRAX = {
    "name" : "HIRAX_32_6",
    "dish_size" : 6.0,
    "t_inst" : 50.0,
    "t_total" : 10000.0,
    "n_dish" : 1024,
    "area_eff" : 1.0,
    "im_type" : "interferometer",
    "base_file" : "curves_IM/baseline_file_HIRAX_6m.txt",
    "base_min" : 6.0,
    "base_max" : 180.0,
    "fsky" : 0.4,
    "omega_sky" : 2000.0
}

class BiPZ(object):
    def __init__(self, redshift, sigma_z=0.03, kFG=0.01, wedge='pb', window='HS', b2=False, IM='SKA'):
        
        self.kmin = 1E-3
        # Photo-z sigz will enter in sigma = sigz(1+z)
        self.sigz = sigma_z
        self.zcen = redshift
        
        # Cosmological quantities
        self.sigma8z = pclass.sigma(8.0/h,self.zcen)
        self.chi  = comov(self.zcen)
        self.HMpc = H(self.zcen)/c
        self.kvec = np.logspace(-5.0, np.log10(250), 1000)
        self.Plin = interp1d(self.kvec/h,np.array([pclass.pk(_k, self.zcen) for _k in self.kvec])*(h**3.0),kind='cubic')
        self.Pnl  = interp1d(self.kvec/h,np.array([pnlclass.pk(_k, self.zcen) for _k in self.kvec])*(h**3.0),kind='cubic')   
        
        # Galaxy properties and survey area
        self.b_g = 0.95/Dz(self.zcen)
        z0    = 0.28
        alpha = 2.0
        beta  = 0.9
        A = beta/(pow(z0,alpha+1)*gamma((alpha+1)/beta))
        self.nbar_arcmin2 = A*pow(self.zcen,alpha)*np.exp(-pow(self.zcen/z0,beta))
        # 3D density in (Mpc/h)^-3
        self.ndens_3D = ( self.nbar_arcmin2/pow((np.pi/180.0)/60.0, 2.0) ) * self.HMpc / pow(self.chi/(1.0+self.zcen), 2.0)
        self.Nk_gg = 1./self.ndens_3D
        self.Area = 13800.0 * pow(np.pi/180,2.0) * self.chi**2.0
                
        # Photo-z uncertainty in Mpc/h
        self.sig_para = self.sigz*(1+self.zcen)/self.HMpc
        
        # HI specifications
        self.IM = IM
        if self.IM == 'SKA':
            HI = SKA
        else:
            HI = HIRAX

        N_dishes  = HI["n_dish"]
        D_dish    = HI["dish_size"]
        N_beams   = 1
        Omega_sky = HI["omega_sky"]
        Omega_tot = Omega_sky * pow(np.pi/180,2.0) # rad
        t_tot     = 10000 * 60 * 60 # seconds
        freq_res  = 15.2*1e3 # Frequency resolution in the Band 1 for SKA1, in Hertz

        # HI Bias
        def b_h_f(z_):
            bHI_0 = 4.3/4.86
            resp = (bHI_0/0.677105)*(0.66655 + 0.17765*z_ + 0.050223*pow(z_,2.0))
            return resp

        self.b_h = b_h_f(self.zcen)
        self.b2 = b2
        
        if self.b2:
            self.bh_2 = 0.412 - 2.143*self.b_h + 0.929*pow(self.b_h,2.0) + 0.008*pow(self.b_h,3.0)
            self.bg_2 = 0.412 - 2.143*self.b_g + 0.929*pow(self.b_g,2.0) + 0.008*pow(self.b_g,3.0)
        else:
            self.bh_2 = 0.0
            self.bg_2 = 0.0

        # Beam
        self.FWHM = 0.21*(1+self.zcen)/D_dish
        self.sigma_beam = self.FWHM/2.355
        self.sig_perp = self.sigma_beam*self.chi


        # Foregrounds
        self.kFG = kFG
        self.window = window
        self.wedge = wedge

        # Mean temperature in K
        Tbright = 1e-3 * (0.055919 + 0.23242*self.zcen - 0.024136*self.zcen**2.0)

        # Noise temperature in K
        def T_sys_SKA(z_):
            # Observed 21cm frequency at z_ (in MHz)
            freq = 1420.4 / (1.0 + z_) 
            # Receiver noise
            Trx = 15 + 30 * pow(freq*1e-3 - 0.75,2.0)
            # Galaxy temperature
            Tgal = 25 * pow(408/freq, 2.75)
            Tspill = 3.0
            Tcmb = 2.73
            return Trx + Tspill + Tcmb + Tgal

        Tsys = T_sys_SKA(self.zcen)

        import noise_21cm as noise21
        if self.IM == 'SKA':
            # Noise power spectrum (in [Mpc/h]^3) - single dish
            self.Nk_hh = (Tsys*self.chi*(1+self.zcen)/Tbright)**(2.0)*Omega_tot / (self.HMpc*1420.*1E6*t_tot*N_dishes*N_beams)
        else:
            # Noise power spectrum (in [mK^2 x (Mpc/h)^3]) - interferometer
            self.k_hh, self.Nk_hh_infm = noise21.get_noisepower_imap(HIRAX, self.zcen)
            # Noise power spectrum (in [(Mpc/h)^3]) - interferometer
            self.Nk_hh = interp1d( self.k_hh,1E-6*self.Nk_hh_infm/pow(Tbright,2.0) , bounds_error=False, fill_value='extrapolate')

        # Compute and spline I_TT
        self._set_ITT()

        # Bispectrum calculation
        self._set_bispectra()

    def phi2_k(self, kpara):
        return np.exp(-(kpara*self.sig_para)**2.0)

    def B_h(self, kperp):
        return np.exp(-0.5*(kperp*self.sig_perp)**2.0)

    def kwedge(self, k_perp):
            
        if self.wedge == 'horizon' or self.wedge == 'hor': 
            self.resp = k_perp*self.chi*self.HMpc/(1+self.zcen)
        elif self.wedge == 'pb':
            self.resp = k_perp*np.sin(self.FWHM/2.0)*self.chi*self.HMpc/(1+self.zcen)
        else:
            self.resp = np.zeros_like(k_perp)
            
        return self.resp   
        
    def B_FG(self, k_para, k_perp, ind=1.):
        alpha_FG  = 1.0
        wedge_FG  = np.heaviside(np.fabs(k_para)-self.kwedge(k_perp),0.0)
        
        if self.window == 'HS_chi' or self.window == 'chi':
            window_FG = np.heaviside(np.fabs(k_para)-2*np.pi/self.chi,0.0)
            resp = alpha_FG * window_FG
        elif self.window == 'HS' or self.window == 'HS_kfg':
            window_FG = np.heaviside(np.fabs(k_para)-self.kFG,0.0)
            resp = alpha_FG * window_FG
        else:
            resp = alpha_FG * ( 1 - np.exp(-(np.fabs(k_para)/self.kFG)**ind) )

        return wedge_FG*resp

    def Pgg(self, k_para, k_perp):
        k = np.sqrt(k_para**2.0+k_perp**2.0)
        pk = self.Pnl(k)
        return self.b_g**2.0*pk+self.Nk_gg

    def Pgh(self, k_para, k_perp):
        k = np.sqrt(k_para**2.0+k_perp**2.0)
        pk = self.Pnl(k)
        bfg = self.B_FG(k_para,k_perp)
        beam = 1.0

        if self.IM == 'SKA':
            beam = self.B_h(k_perp)
            
        return self.b_h*self.b_g*beam*bfg*pk 

    def Phh(self, k_para, k_perp):
        k = np.sqrt(k_para**2.0+k_perp**2.0)
        pk = self.Pnl(k)
        bfg = self.B_FG(k_para,k_perp)

        if self.IM == 'SKA':
            beam = self.B_h(k_perp)
            self.resp = (self.b_h*bfg*beam)**2.0*pk+self.Nk_hh
        else:
            beam = 1.0
            self.resp = (self.b_h*bfg*beam)**2.0*pk+self.Nk_hh(k_perp)
            
        return self.resp
        
    def _set_ITT(self):
        def integ_ITT(kpara, kperp, w_R=True):
            pgg = self.Pgg(kpara, kperp)
            R = 1
            if w_R:
                pgh = self.Pgh(kpara, kperp)
                phh = self.Phh(kpara, kperp)
                R = 1 - pgh**2.0/(pgg*phh)
            return self.phi2_k(kpara)*pgg*R/(2.0*np.pi)

        kpara_sample = np.linspace(0, 3.0/self.sig_para, 40)
        kperps = np.geomspace(1E-4, 1E2, 256)
        ITT1 = 1./np.array([simps(2.0*integ_ITT(kpara_sample, kperp, w_R=True), x=kpara_sample)
                            for kperp in kperps])

        self.ITTi = interp1d(np.log(kperps), np.log(ITT1), bounds_error=False, fill_value='extrapolate')

    def ITT(self, kperp):
        return np.exp(self.ITTi(np.log(kperp)))
        
    def get_Nmodes_Pk(self, nsigma_para=5,
                      npara_per_sigma=10,
                      kmax=1.0,
                      nk_perp_per_decade=16):
        def int_Nmodes(kpara, kperp):
            pgh = self.Pgh(kpara, kperp)
            phh = self.Phh(kpara, kperp)
            x2 = (kpara * self.sig_para)**2.0
            phi2 = np.exp(-x2)
            k = np.sqrt(kpara**2.0+kperp**2.0)
            pk = self.Pnl(k)
            pgg = self.b_g**2.0 * pk*phi2+self.Nk_gg
            return kperp**2.0 * phi2*pgh**2.0/(phh*pgg)

        nk_para = nsigma_para*npara_per_sigma
        kpara_sample = np.linspace(0, min(kmax, nsigma_para/self.sig_para), nk_para)
        nk_perp = int(np.log10(kmax/self.kmin)*nk_perp_per_decade)
        kperps = np.geomspace(self.kmin, kmax, nk_perp)
        nmodes_perp = np.array([simps(int_Nmodes(kpara_sample, kperp), x=kpara_sample)
                                for kperp in kperps])
        nmodes = simps(nmodes_perp, x=np.log(kperps))/(2.0*np.pi**2.0)
        
        return nmodes

    def get_VarP(self, is_sigma=False, nsigma_para=3, nk_para=32, kmax=1.0, nk_perp=64):
        def int_FisherP(kpara, kperp):
            I_TT = self.ITT(kperp)
            pgh = self.Pgh(kpara, kperp)
            phh = self.Phh(kpara, kperp)
            x2 = (kpara * self.sig_para)**2.0
            if is_sigma:
                return kperp**2.0*I_TT*kpara**2.0*x2*np.exp(-x2)*pgh**2.0/(phh*2.0*np.pi)
            else:
                return kperp**2.0*I_TT*kpara**2.0*np.exp(-x2)*pgh**2.0/(phh*2.0*np.pi)

        kpara_sample = np.linspace(0, nsigma_para/self.sig_para, nk_para)
        kperps = np.geomspace(1E-3, kmax, nk_perp)
        int_FisherP_perp = np.array([simps(2.0*int_FisherP(kpara_sample, kperp), x=kpara_sample)
                                     for kperp in kperps])
        varP = 1./(self.Area*simps(int_FisherP_perp, x=np.log(kperps))/(2.0*np.pi))
        return varP

    def _set_bispectra(self):
        
        self.kvec1 = np.geomspace(1E-4, 250, 5000)
        self.intersect = interp1d( (self.Plin(self.kvec1) * (self.kvec1)**3.0) / (2.0*pow(np.pi,2.0)), self.kvec1 )
        self.k_nl = self.intersect(1.0)
        self.n_slope = np.gradient(np.log(self.Plin(self.kvec1)),np.log(self.kvec1))
             
        dndk         = np.diff(self.n_slope,n=1)
        self.cp      = []
        self.n_at_cp = []
        self.mid_pts = []
        self.loc_mid = []

        # Find critical points (dx/dy = 0)
        for i in range(1,len(self.kvec1)-1):
            if (dndk[i-1]<0) != (dndk[i-1+1]<0):
                self.cp.append(i)
                self.n_at_cp.append(self.n_slope[i])

        # Find middle points
        i = 0
        while i != len(self.cp)-3:
            if i == 0:
                tmp_max = max(self.n_slope)
                tmp_min = self.n_slope[self.cp[i]]
                self.mid_pts.append((tmp_max + tmp_min)/2.0)
                i=1
            else:
                tmp_max = self.n_slope[self.cp[i]]
                tmp_min = self.n_slope[self.cp[i+1]]
                self.mid_pts.append((tmp_max + tmp_min)/2.0)
                i+=2
                
        # Obtain k-values for the 
        self.fun_k = []
        for i in range(len(self.cp)-1):
            if i == 0:
                rng = range(self.cp[i]+1)
                self.fun_k.append(interp1d( self.n_slope[rng],self.kvec1[rng],kind='cubic',fill_value='extrapolate', bounds_error=False ))
            elif i%2 == 1: 
                rng = range(self.cp[i], self.cp[i+1]+1)
                self.fun_k.append(interp1d( self.n_slope[rng],self.kvec1[rng],kind='cubic',fill_value='extrapolate', bounds_error=False ))

        for i in range(len(self.mid_pts)):
            self.loc_mid.append(self.fun_k[i](self.mid_pts[i]).tolist())

        # Obtain mid_pts and their respective location
        self.intersect = interp1d( self.kvec1,self.n_slope, kind='cubic',fill_value='extrapolate', bounds_error=False )
        del self.mid_pts
        self.mid_pts = self.intersect(self.loc_mid).tolist()

        self.low = np.where(self.kvec1<5e-3)
        self.high = np.where(self.kvec1>1)
        self.interp_k = np.concatenate((self.kvec1[self.low],self.loc_mid,self.kvec1[self.high]))
        self.interp_n = np.concatenate((self.n_slope[self.low],self.mid_pts,self.n_slope[self.high]))
        
        a1 = 0.484
        a2 = 3.740
        a3 = -0.849
        a4 = 0.392
        a5 = 1.013
        a6 = -0.575
        a7 = 0.128
        a8 = -0.722
        a9 = -0.926
        s8 = self.sigma8z

        def Q3_f(n):
            return (4.0-pow(2.0, n))/(1.0+pow(2.0, n+1.0))

        def a_f(k):
            n = self.n_smooth(k)
            q = k/self.k_nl
            den = pow(q*a1, n+a2)
            num = s8**a6*np.sqrt(0.7*Q3_f(n))*den
            return (1.0+num)/(1.0+den)

        def b_f(k):
            n = self.n_smooth(k)
            q = k/self.k_nl
            num = 0.2*a3*(n+3)*pow(q*a7, n+3+a8)
            den = pow(q*a7, n+3.5+a8)
            return (1+num)/(1+den)

        def c_f(k):
            n = self.n_smooth(k)
            q = k/self.k_nl
            num = 4.5*a4*pow(q*a5, n+3+a9)/(1.5+(n+3)**4.0)
            den = pow(q*a5, n+3.5+a9)
            return (1.0+num)/(1.0+den)

        self.a_fun = interp1d(self.kvec1, a_f(self.kvec1), kind='linear',
                              fill_value='extrapolate', bounds_error=False)
        self.b_fun = interp1d(self.kvec1, b_f(self.kvec1), kind='linear',
                              fill_value='extrapolate', bounds_error=False)
        self.c_fun = interp1d(self.kvec1, c_f(self.kvec1), kind='linear',
                              fill_value='extrapolate', bounds_error=False)
                              
    def n_smooth(self, k):
        nsmooth = interp1d( self.interp_k,self.interp_n, kind='cubic',fill_value='extrapolate', bounds_error=False)
        return nsmooth(k)
    
    def F2_eff(self, ki, kj, kk):
        cos_ij = (kk**2.0-ki**2.0-kj**2.0)/(2.0*ki*kj)
        ai = self.a_fun(ki)
        aj = self.a_fun(kj)
        bi = self.b_fun(ki)
        bj = self.b_fun(kj)
        ci = self.c_fun(ki)
        cj = self.c_fun(kj)
        return (5.0*ai*aj/7.0 +
                0.5*cos_ij*(ki/kj+kj/ki)*bi*bj +
                2.0*cos_ij**2.0*ci*cj/7.0)

    def get_VarB(self, kmax=1., is_sigma=False,
                 nkperp_hi=32, nkperp_lo=16, nsigma_perp=3,
                 nkpar_hi=32, nkpar_lo=32, nsigma_par=3, verbose=False):
        def int_FisherB(k1_para, k2_para, k1_perp, k2_perp, k3_perp):
            # Integrand of Fisher matrix element
            # Integration variables are k1_para, log(k2_para), and log(k1/2/3_perp)
            if (k3_perp <= np.fabs(k1_perp-k2_perp)) or (k3_perp >= k1_perp+k2_perp):
                return np.zeros_like(k1_para)
            k3_para = -(k1_para + k2_para)
            k1 = np.sqrt(k1_para**2.0+k1_perp**2.0)
            k2 = np.sqrt(k2_para**2.0+k2_perp**2.0)
            k3 = np.sqrt(k3_para**2.0+k3_perp**2.0)
            
            AT = 0.5*np.sqrt(2.0*(k1_perp**2.0*k2_perp**2.0+
                                k2_perp**2.0*k3_perp**2.0+
                                k3_perp**2.0*k1_perp**2.0)-
                             k1_perp**4.0-k2_perp**4.0-k3_perp**4.0)
            I_TT = self.ITT(k1_perp)
            pre_perp = I_TT*(k1_perp*k2_perp*k3_perp/np.pi)**2.0/AT
            phh_a = self.Phh(k2_para, k3_perp)
            phh_b = self.Phh(k3_para, k2_perp)
            F123 = self.F2_eff(k1, k2, k3)
            F231 = self.F2_eff(k2, k3, k1)
            F312 = self.F2_eff(k3, k1, k2)
            
            beams = 1.0
            if self.IM == 'SKA':
                beams = self.B_h(k2_perp)*self.B_h(k3_perp)

            beams_HI = self.B_FG(k2_para,k2_perp)*self.B_FG(k3_para,k3_perp)*beams
            pk1 = self.Pnl(k1)
            pk2 = self.Pnl(k2)
            pk3 = self.Pnl(k3)
            if self.b2:
                pk1L = self.Plin(k1)
                pk2L = self.Plin(k2)
                pk3L = self.Plin(k3)
                Bbias = self.b_g*self.b_h*self.bh_2 * pk1L*pk2L + self.b_g*self.bh_2*self.b_h * pk1L*pk3L + self.bg_2*self.b_h*self.b_h * pk2L*pk3L
            else:
                Bbias = 0.0
            
            Bgrav = 2.0*(F123*pk1*pk2+F231*pk2*pk3+F312*pk3*pk1)
            Bghh = beams_HI * (self.b_g*(self.b_h**2.0)*Bgrav + Bbias)
            x2 = (k1_para*self.sig_para)**2.0
            phi2 = (k1_para/(2.0*np.pi))**2.0*np.exp(-x2)
            if is_sigma:
                return pre_perp*k2_para*x2*phi2*Bghh**2.0/(phh_a*phh_b)
            else:
                return pre_perp*k2_para*phi2*Bghh**2.0/(phh_a*phh_b)

        k1_perps = np.geomspace(1E-3, kmax, nkperp_hi)
        k2_perps = np.geomspace(1E-3, kmax, nkperp_lo)
        k3_perps = np.geomspace(1E-3, kmax, nkperp_lo)
        k1_pars = np.linspace(-3/self.sig_para, 3/self.sig_para, nkpar_lo)
        k2_pars = np.geomspace(1E-3, kmax, nkpar_hi)

        integ_k2par = np.zeros(len(k2_pars))
        integ_perp = np.zeros([len(k1_perps), len(k2_perps), len(k3_perps)])

        for i1, k1 in enumerate(k1_perps):
            if verbose:
                print(i1)
            for i2, k2 in enumerate(k2_perps):
                for i3, k3 in enumerate(k3_perps):
                    for j2, k2p in enumerate(k2_pars):
                        integ = int_FisherB(k1_pars, k2p, k1, k2, k3)
                        integ_k2par[j2] = simps(integ, x=k1_pars)
                    # Factor 2 to account for negative k2 values
                    integ_perp[i1, i2, i3] = 2.0*simps(integ_k2par, x=np.log(k2_pars))

        integ_12 = simps(integ_perp, x=np.log(k3_perps), axis=-1)
        integ_1 = simps(integ_12, x=np.log(k2_perps), axis=-1)
        VarBr = 1.0/(self.Area*simps(integ_1, x=np.log(k1_perps))/(24*np.pi))
        return VarBr
