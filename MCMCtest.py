import orbfits
import numpy as np
import os
from psrqpy import QueryATNF
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from multiprocessing import Pool
from astropy.coordinates import SkyCoord

os.environ["OMP_NUM_THREADS"] = "1"


names=list(f[6:-4] for f in os.listdir('./binarytimestamps'))
names_cat=list(names)
names_cat[0]='J1939+2134'
names_cat[6]=names[6]+'+2551'
names.remove(names[6])
names_cat.remove(names_cat[6])
query=QueryATNF(psrs=names_cat)
psrs = query.get_pulsars()

Test_source = 'J1643-1224'
srce=SkyCoord.from_name('PSR %s' %Test_source)

PSR = psrs[Test_source]
times_str = np.load('./binarytimestamps/times_%s.npy' %
                    Test_source).astype(str)
times = Time(times_str)

##Knowns
dp = PSR.DIST_DM1 * u.kpc
Om_peri = PSR.OM * u.deg
Om_peri_dot = PSR.OMDOT * u.deg / u.year
A1 = PSR.A1 * u.s * const.c
Ecc = PSR.ECC
Pb = PSR.PB * u.day
if str(PSR.T0) == '--':
    TASC = Time(PSR.TASC, format='mjd')
    T0 = Time(brentq(T0_find,
                     TASC.mjd, (TASC + Pb).mjd,
                     args=(TASC, -Om_peri, Pb, Ecc)),
              format='mjd')
else:
    T0 = Time(PSR.T0, format='mjd')
pm_ra = (PSR.PMRA * u.mas / u.year).to_value(u.rad / u.s) / u.s
pm_dec = (PSR.PMDec * u.mas / u.year).to_value(u.rad / u.s) / u.s
f0 = 1 * u.GHz

##Unknowns
Om_orb = 38.2 * u.deg
Om_scr = 0 * u.deg
inc = 20 * u.deg
ds = dp / 2

a = np.abs(A1 / np.sin(inc))
eta_data = orbfits.eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                   dp, ds, f0, pm_ra, pm_dec)

sigma = eta_data / 10

eta_noisy = eta_data + np.random.normal(0, 1, eta_data.shape[0])*sigma

lwrs=np.array([0,0,0,0])
uprs=np.array([360,360,360,dp.to_value(u.kpc)])

pool=Pool()

ndim, nwalkers = 4, 40
pos = [np.random.uniform(lwrs,uprs) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, orbfits.lnprob, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),pool=pool)

sampler.run_mcmc(pos, 1000, progress=True)