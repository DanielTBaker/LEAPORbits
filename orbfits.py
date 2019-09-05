from astropy.coordinates import solar_system_ephemeris, EarthLocation, SkyCoord
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import SphericalRepresentation
from astropy.time import Time
import astropy.units as u
import numpy as np
import astropy.constants as const

def E_proj_vel(srce,T):
    srce_cart=srce.transform_to('barycentrictrueecliptic').represent_as('cartesian')
    earth_posvel=get_body_barycentric_posvel('earth', T)

    srce_dra=SkyCoord(ra=srce.ra+1e-7*u.deg,dec=srce.dec)
    srce_dra_cart=srce_dra.transform_to('barycentrictrueecliptic').represent_as('cartesian')
    unit_dra=srce_dra_cart-srce_cart
    unit_dra=unit_dra/np.sqrt(unit_dra.x**2+unit_dra.y**2+unit_dra.z**2)

    E_V_ra=unit_dra.dot(earth_posvel[1]).to(u.km/u.s)

    srce_ddec=SkyCoord(ra=srce.ra,dec=srce.dec+1e-7*u.deg)
    srce_ddec_cart=srce_ddec.transform_to('barycentrictrueecliptic').represent_as('cartesian')
    unit_ddec=srce_ddec_cart-srce_cart
    unit_ddec=unit_ddec/np.sqrt(unit_ddec.x**2+unit_ddec.y**2+unit_ddec.z**2)

    E_V_dec=unit_ddec.dot(earth_posvel[1]).to(u.km/u.s)
    return(E_V_ra,E_V_dec)

def orb_proj_vel(times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,
                       inc):
    Om_peri_cur = Om_peri + Om_peri_dot * (times - T0)
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
    V_x2 = V_x * np.cos(Om_peri_cur) - V_y * np.sin(Om_peri_cur)
    V_y2 = V_x * np.sin(Om_peri_cur) + V_y * np.cos(Om_peri_cur)
    V_y3 = V_y2 * np.cos(inc)
    V_ra = V_x2*np.cos(Om_orb) - V_y3*np.sin(Om_orb)
    V_dec = V_x2*np.sin(Om_orb) + V_y3*np.cos(Om_orb)
    return (V_ra.to(u.km/u.s), V_dec.to(u.km/u.s))

def eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,inc,dp,ds,f0,pm_ra,pm_dec):
    dps = dp - ds
    Orb_ra,Orb_dec= orb_proj_vel(times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,
        inc)
    V_P_ra=pm_ra*dp+Orb_ra
    V_P_dec=pm_dec*dp+Orb_dec
    V_E_ra,V_E_dec=E_proj_vel(srce,times)
    V_E_par = V_E_ra*np.cos(Om_scr) + V_E_dec*np.sin(Om_scr)
    V_P_par = V_P_ra*np.cos(Om_scr) + V_P_dec*np.sin(Om_scr)
    Veff = V_E_par + (ds / dp) * V_P_par

    eta = .5 * (ds * dp / dps) * (const.c / ((Veff)**2)) / f0**2
    return(eta.to(u.ms/u.mHz**2))

def eta_sol(times,Ecc,Om_scr,dp,ds,f0,pm_ra,pm_dec):
    dps = dp - ds
    V_P_ra=pm_ra*dp
    V_P_dec=pm_dec*dp
    V_E_ra,V_E_dec=E_proj_vel(SkyCoord.from_name('PSR %s' %names_cat[srce_idx]),times)
    V_E_par = V_E_ra*np.cos(Om_scr) + V_E_dec*np.sin(Om_scr)
    V_P_par = V_P_ra*np.cos(Om_scr) + V_P_dec*np.sin(Om_scr)
    Veff = V_E_par + (ds / dp) * V_P_par

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
    return(0.0)
   
def lnprob(theta,data,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs):
    Om_orb,Om_scr,inc,ds = theta
    Om_orb*=u.deg
    Om_scr*=u.deg
    inc*=u.deg
    ds*=u.kpc
    a = np.abs(A1 / np.sin(inc))
    lp=lnprior(theta,lwrs,uprs)
    if not np.isfinite(lp):
        return -np.inf
    inv_sigma2 = 1.0/sigma**2
    model=eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr,inc,dp,ds,f0,pm_ra,pm_dec)
    
    lp2 = -0.5*(np.sum((data-model).value**2*inv_sigma2.value - np.log(inv_sigma2.value)))
    return(lp+lp2)