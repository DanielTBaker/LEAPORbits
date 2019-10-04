from astropy.coordinates import solar_system_ephemeris, EarthLocation, SkyCoord
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import SphericalRepresentation
from astropy.time import Time
import astropy.units as u
import numpy as np
import astropy.constants as const
from psrqpy import QueryATNF

##Calculate ra and dec components of Earth's velocity on the sky
def E_proj_vel(srce,T):
    ##Get source location
    srce_cart=srce.represent_as('cartesian')
    ##Get Earth velocity
    earth_posvel=get_body_barycentric_posvel('earth', T)

    ##Find unit vector in ra direction
    srce_dra=SkyCoord(ra=srce.ra+1e-7*u.deg,dec=srce.dec,frame='icrs')
    srce_dra_cart=srce_dra.represent_as('cartesian')
    unit_dra=srce_dra_cart-srce_cart
    unit_dra=unit_dra/np.sqrt(unit_dra.x**2+unit_dra.y**2+unit_dra.z**2)

    E_V_ra=unit_dra.dot(earth_posvel[1]).to(u.km/u.s)

    ##Find unit vector in dec direction
    srce_ddec=SkyCoord(ra=srce.ra,dec=srce.dec+1e-7*u.deg,frame='icrs')
    srce_ddec_cart=srce_ddec.represent_as('cartesian')
    unit_ddec=srce_ddec_cart-srce_cart
    unit_ddec=unit_ddec/np.sqrt(unit_ddec.x**2+unit_ddec.y**2+unit_ddec.z**2)

    E_V_dec=unit_ddec.dot(earth_posvel[1]).to(u.km/u.s)
    return(E_V_ra,E_V_dec)

##Calculate ra and dec components of orbital velocity
def orb_proj_vel(times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,
                       inc):
    ##Current location of periastron
    Om_peri_cur = Om_peri + Om_peri_dot * (times - T0)
    ##Find x and y components of velocity in orbital frame
    M = ((times - T0) * (2 * np.pi * u.rad / Pb)).to_value(u.rad)
    Mdot=(2 * np.pi * u.rad / Pb)
    TA = M + (2 * Ecc -
              (Ecc**3) / 4) * np.sin(M) + (5 / 4) * (Ecc**2) * np.sin(
                  2 * M) + (13 / 12) * (Ecc**3) * np.sin(3 * M)
    TAdot=(1+(2 * Ecc -
              (Ecc**3) / 4) * np.cos(M) + 2*(5 / 4) * (Ecc**2) * np.cos(
                  2 * M) + 3*(13 / 12) * (Ecc**3) * np.cos(3 * M))*Mdot
    R = a * (1 - Ecc**2) / (1 + Ecc * np.cos(TA))
    dRdTA = np.sin(TA)*Ecc*(R**2)/(a*(1-Ecc**2))
    x = R * np.cos(TA)
    y = R * np.sin(TA)
    V_x =(dRdTA*np.cos(TA) - y)*TAdot/u.rad
    V_y =(dRdTA*np.sin(TA) + x)*TAdot/u.rad
    ##Rotate such that x2 is along the line of nodes
    V_x2 = V_x * np.cos(Om_peri_cur) - V_y * np.sin(Om_peri_cur)
    V_y2 = V_x * np.sin(Om_peri_cur) + V_y * np.cos(Om_peri_cur)
    ##Project V_y onto the sky
    V_y3 = V_y2 * np.cos(inc)
    ##Rotate to RA and Dec
    V_ra = V_x2*np.cos(Om_orb) - V_y3*np.sin(Om_orb)
    V_dec = V_x2*np.sin(Om_orb) + V_y3*np.cos(Om_orb)
    return (V_ra.to(u.km/u.s), V_dec.to(u.km/u.s))

##Calculate eta from model
def eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,inc,dp,ds,f0,pm_ra,pm_dec):
    dps = dp - ds
    Orb_ra,Orb_dec= orb_proj_vel(times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,
        inc)
    V_P_ra=pm_ra*dp+Orb_ra
    V_P_dec=pm_dec*dp+Orb_dec
    V_E_ra,V_E_dec=E_proj_vel(srce,times)
    V_E_par = V_E_ra*np.cos(Om_scr) + V_E_dec*np.sin(Om_scr)
    V_P_par = V_P_ra*np.cos(Om_scr) + V_P_dec*np.sin(Om_scr)
    Veff = V_E_par + (ds / dps) * V_P_par

    eta = .5 * (ds * dp / dps) * (const.c / ((Veff)**2)) / f0**2
    return(eta.to(u.ms/u.mHz**2))

def T0_find(T0,TA,nu0,Pb,Ecc):
    M=2*np.pi*u.rad*(TA-Time(T0,format='mjd'))/Pb
    nu=M.value+(2*Ecc-(Ecc/2)**2)*np.sin(M)+(5/4)*(Ecc**2)*np.sin(2*M)+(13/12)*(Ecc**3)*np.sin(3*M)
    return(nu-nu0.to_value(u.rad))

def solver(X,data,sigma,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1):
    Om_orb,Om_scr,inc,ds = X
    Om_orb*=u.deg
    Om_scr*=u.deg
    inc*=u.deg
    ds*=u.kpc
    a=np.abs(A1/np.sin(inc))
    eta_model=eta_orb(times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,inc,dp,ds,f0,pm_ra,pm_dec)
    val=np.sum(((eta_model-data)/sigma)**2)
    print(X,' : ', val)
    return(val)
    
def lnprior(theta,lwrs,uprs):
    if (theta-lwrs).min()<0:
        return(-np.inf)
    if (theta-uprs).max()>0:
        return(-np.inf)
    return(np.log(np.sin(theta[-2]*u.deg).value/2))
   
def lnprob(theta,data,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs):
    Om_orb,Om_scr,inc,ds = theta
    Om_orb*=u.deg
    Om_scr*=u.deg
    inc*=u.deg
    ds*=u.kpc
    a = np.abs(A1 / np.sin(inc))
    lp=lnprior(theta,lwrs,uprs)
    if not np.isfinite(lp):
        return(-np.inf)
    inv_sigma2 = 1.0/sigma**2
    model=eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,inc,dp,ds,f0,pm_ra,pm_dec)
    if sigma.max()==np.inf:
        if (data[sigma==np.inf]-model[sigma==np.inf]).min()<0:
            return(-np.inf)
    lp2 = -0.5*(np.sum((data[sigma<np.inf]-model[sigma<np.inf]).value**2*inv_sigma2[sigma<np.inf].value - np.log(inv_sigma2[sigma<np.inf].value)))
    return(lp2)

def bound_prior(x,lower,upper):
    if x>lower and x<upper:
        return(0.0)
    else:
        return(-np.inf)

class parameter:
    def  __init__(self,value,lnprior=lambda x: 0,fixed=True):
        self.fixed=fixed
        self.lnprior=lnprior
        self.val=value
class PSR_fit:
    def __init__(self,name,data=None,sigma=None):
        self.name=name

        if not os.path.isfile('%s_params.npy' %self.name):
            try:
                query=QueryATNF(psrs=[self.name])
                psrs = query.get_pulsars()
                cat=True
            except:
                print('psrqpy not available',flush=True)
                sys.exit()
            PSR = psrs[self.name]
            
            self.dp = parameter(PSR.DIST_DM1 * u.kpc)
            if not PSR.DIST_DM1_ERR==None:
                sig = PSR.DIST_DM1_ERR
                self.dp.lnprior = lambda x: np.exp(.5*((x-self.dp.val.value)/sig)**2)/np.sqrt(2*np.pi*err**2)
                self.dp.fixed = False
            
            self.Om_peri = parameter(PSR.OM * u.deg)
            if not PSR.OM_ERR==None:
                sig = PSR.OM_ERR
                self.Om_peri.lnprior = lambda x: np.exp(.5*((x-self.Om_peri.val.value)/sig)**2)/np.sqrt(2*np.pi*err**2)
                self.Om_peri.fixed = False

            self.Om_peri_dot = parameter(PSR.OMDOT * u.deg / u.year)
            if not PSR.OMDOT_ERR==None:
                sig = PSR.OMDOT_ERR
                self.Om_peri_dot.lnprior = lambda x: np.exp(.5*((x-self.Om_peri_dot.val.value)/sig)**2)/np.sqrt(2*np.pi*err**2)
                self.Om_peri_dot.fixed = False

            self.A1 = parameter(PSR.A1 * u.s * const.c)
            if not PSR.A1_ERR==None:
                sig = PSR.A1_ERR * const.c.value
                self.A1.lnprior = lambda x: np.exp(.5*((x-self.A1.val.value)/sig)**2)/np.sqrt(2*np.pi*err**2)
                self.A1.fixed = False
            
            self.Ecc = parameter(PSR.ECC)
            if not PSR.ECC_ERR==None:
                sig = PSR.ECC_ERR
                self.Ecc.lnprior = lambda x: np.exp(.5*((x-self.A1.val)/sig)**2)/np.sqrt(2*np.pi*err**2)
                self.Ecc.fixed = False
            
            self.Pb = parameter(PSR.PB * u.day)
            if not PSR.PB_ERR==None:
                sig = PSR.PB_ERR
                self.Pb.lnprior = lambda x: np.exp(.5*((x-self.Pb.val.value)/sig)**2)/np.sqrt(2*np.pi*err**2)
                self.Pb.fixed = False
            if str(PSR.T0) == '--':
                TASC = Time(PSR.TASC, format='mjd')
                self.T0 = parameter(Time(brentq(orbfits.T0_find,
                                TASC.mjd, (TASC + Pb).mjd,
                                args=(TASC, -Om_peri, Pb, Ecc)),
                        format='mjd'))
                if not PSR.TASC_ERR==None:
                    sig = PSR.TASC_ERR
                    self.T0.lnprior = lambda x: np.exp(.5*((x-self.T0.val.mjd)/sig)**2)/np.sqrt(2*np.pi*err**2)
                    self.T0.fixed = False 
            else:
                self.T0 = parameter(Time(PSR.T0, format='mjd'))
                if not PSR.T0_ERR==None:
                    sig = PSR.PB_ERR
                    self.T0.lnprior = lambda x: np.exp(.5*((x-self.T0.val.mjd)/sig)**2)/np.sqrt(2*np.pi*err**2)
                    self.T0.fixed = False

            
            self.pm_ra = parameter((PSR.PMRA * u.mas / u.year).to_value(u.rad / u.s) / u.s)
            if not PSR.PMRA_ERR==None:
                sig = (PSR.PMRA_ERR*u.mas/u.year).to_value(u.rad/u.s)
                self.pm_ra.lnprior = lambda x: np.exp(.5*((x-self.pm_ra.val.value)/sig)**2)/np.sqrt(2*np.pi*err**2)
                self.pm_ra.fixed = False
            
            self.pm_dec = parameter((PSR.PMDec * u.mas / u.year).to_value(u.rad / u.s) / u.s)
            if not PSR.PMDec_ERR==None:
                sig = (PSR.PMDec_ERR*u.mas/u.year).to_value(u.rad/u.s)
                self.pm_dec.lnprior = lambda x: np.exp(.5*((x-self.pm_dec.val.value)/sig)**2)/np.sqrt(2*np.pi*err**2)
                self.pm_dec.fixed = False
            self.srce=SkyCoord.from_name('PSR %s' %Test_source)
            np.save('%s_params.npy' %Test_source,np.array([dp.value, Om_peri.value, Om_peri_dot.value, A1.value, Ecc, Pb.value, T0.mjd,pm_ra.value,pm_dec.value,self.srce.ra.to_value(u.deg),self.srce.dec.to_value(u.deg)]))

        self.Om_orb=parameter(180,lambda x: bound_prior(x,0,360))
        self.Om_scr=parameter(0,lambda x: bound_prior(x,-90,90),fixed=False)
        self.inc=parameter(45,lambda x: bound_prior(x,0,1),fixed=False)
        self.s=parameter(.5,lambda x: bound_prior(x,0,1),fixed=False)
        self.data=data
        self.sigma=sigma
    
