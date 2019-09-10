import orbfits
import numpy as np
import os
from psrqpy import QueryATNF
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
import argparse
import pickle as pkl
import sys
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(description='Test Orbit Recovery')
parser.add_argument('-ns',type=int,default=2000,help='Number of Samples')
parser.add_argument('-nw',type=int,default=10,help='Number of Walkers')
parser.add_argument('-oo',type=float,default=0,help='Omega Orbit')
parser.add_argument('-os',type=float,default=0,help='Omega Screen')
parser.add_argument('-i',type=float,default=45,help='Inclination')
parser.add_argument('-s',type=float,default=.5,help='Fractional Screen Distance')
parser.add_argument('-th',type=int,default=80,help='Number of Threads')
parser.add_argument('-nT',type=int,default=8,help='Number of Temperatures')

args=parser.parse_args()
print('Import Complete',flush=True)

print('Load and Query',flush=True)
names=list(f[6:-4] for f in os.listdir('./binarytimestamps'))
names_cat=list(names)
names_cat[0]='J1939+2134'
names_cat[6]=names[6]+'+2551'
names.remove(names[6])
names_cat.remove(names_cat[6])


print('Select Source',flush=True)
Test_source = 'J1643-1224'

print('Define Simulation Parameters',flush=True)
times_str = np.load('./binarytimestamps/times_%s.npy' %
                    Test_source).astype(str)
times = Time(times_str)

##Knowns
if not os.path.isfile('%s_params.npy' %Test_source):
    try:
        query=QueryATNF(psrs=names_cat)
        psrs = query.get_pulsars()
        cat=True
    except:
        print('psrqpy not available',flush=True)
        sys.exit()
    PSR = psrs[Test_source]
    dp = PSR.DIST_DM1 * u.kpc
    Om_peri = PSR.OM * u.deg
    Om_peri_dot = PSR.OMDOT * u.deg / u.year
    A1 = PSR.A1 * u.s * const.c
    Ecc = PSR.ECC
    Pb = PSR.PB * u.day
    if str(PSR.T0) == '--':
        TASC = Time(PSR.TASC, format='mjd')
        T0 = Time(brentq(orbfits.T0_find,
                        TASC.mjd, (TASC + Pb).mjd,
                        args=(TASC, -Om_peri, Pb, Ecc)),
                format='mjd')
    else:
        T0 = Time(PSR.T0, format='mjd')
    pm_ra = (PSR.PMRA * u.mas / u.year).to_value(u.rad / u.s) / u.s
    pm_dec = (PSR.PMDec * u.mas / u.year).to_value(u.rad / u.s) / u.s
    srce=SkyCoord.from_name('PSR %s' %Test_source)
    np.save('%s_params.npy' %Test_source,np.array([dp.value, Om_peri.value, Om_peri_dot.value, A1.value, Ecc, Pb.value, T0.mjd,pm_ra.value,pm_dec.value,srce.ra.to_value(u.deg),srce.dec.to_value(u.deg)]))
else:
    dp, Om_peri, Om_peri_dot, A1, Ecc, Pb, T0,pm_ra,pm_dec,ra,dec = np.load('%s_params.npy' %Test_source)
    dp*=u.kpc
    Om_peri*=u.deg
    Om_peri_dot*=u.deg/u.year
    A1*=u.m
    Pb*=u.day
    pm_ra/=u.s
    pm_dec/=u.s
    T0=Time(T0,format='mjd')
    ra*=u.deg
    dec*=u.deg
    srce=SkyCoord(ra=ra,dec=dec)
    
f0 = 1 * u.GHz

##Unknowns
Om_orb = np.mod(args.oo,360) * u.deg
Om_scr = (np.mod(args.os+90,180)-90) * u.deg
inc = np.mod(args.i,90) * u.deg
ds = dp *args.s
print('Fixed Parameters',flush=True)
print('Simulate Data',flush=True)
a = np.abs(A1 / np.sin(inc))
eta_data = orbfits.eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                   dp, ds, f0, pm_ra, pm_dec)

sigma = eta_data / 10

eta_noisy = eta_data + np.random.normal(0, 1, eta_data.shape[0])*sigma

lwrs=np.array([0,-90,0,0])
uprs=np.array([360,90,90,dp.to_value(u.kpc)])



ndim, nwalkers, nthreads, ntemps = 4, args.nw, args.th, args.nT

def PT_func(theta):
   return(orbfits.lnprob(theta,eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs))

def lnprior(theta):
    return 0.0

pos = np.random.uniform(0,1,(ntemps,nwalkers,ndim))*(uprs-lwrs)[np.newaxis,np.newaxis,:]+lwrs[np.newaxis,np.newaxis,:]


sampler=emcee.PTSampler(ntemps, nwalkers, ndim, PT_func, lnprior,threads=nthreads)

print('Start Burn',flush=True)
runs=0
for p, lnprob, lnlike in sampler.sample(pos, iterations=min((1000,args.ns//2))):
    runs+=1
    if np.mod(runs,args.ns//10)==0:
        print('%s/%s Complete' %(runs,args.ns),flush=True)
print('Burn Complete',flush=True)
for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                           lnlike0=lnlike,
                                           iterations=args.ns-200):
    runs+=1
    if np.mod(runs,args.ns//10)==0:
        print('%s/%s Complete' %(runs,args.ns),flush=True)

print('Start Plotting',flush=True)
samples = sampler.chain[0,:, min((1000,args.ns//2)):, :].reshape((-1, ndim))

para_names=np.array([r'$\Omega_{orb}$',r'$\Omega_{scr}$',r'$i$',r'$D_s$'])
para_names_file=np.array(['OmOrb','OmScr','i','Ds'])
reals=np.array([Om_orb.value,Om_scr.value,inc.value,ds.value])

with PdfPages('PT_Results.pdf') as pdf:
    fig = corner.corner(samples, labels=para_names,
                      truths=reals,quantiles=[.16, .50, .84],show_titles=True)
    pdf.savefig()
    plt.close()
    for i in range(ndim):
        fig,axes=plt.subplots(ncols=1,nrows=ntemps,figsize=(4,4*ntemps))
        fig.suptitle(para_names[i])
        for k in range(ntemps):
            for l in range(nwalkers):
                axes[k].plot(sampler.chain[k,l,:,i])
        pdf.savefig()
        plt.close()
    times_curve=Time(np.linspace(times.min().mjd,times.max().mjd,10000),format='mjd')
    eta_fit = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, samples[:,0].mean()*u.deg, samples[:,1].mean()*u.deg, samples[:,2].mean()*u.deg,
                   dp, samples[:,3].mean()*u.kpc, f0, pm_ra, pm_dec)
    eta_real = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                   dp, ds, f0, pm_ra, pm_dec)

    plt.figure()
    plt.plot_date(times.plot_date,eta_noisy,label='Data')
    plt.plot_date(times_curve.plot_date,eta_real,'-',label='Real')
    plt.plot_date(times_curve.plot_date,eta_fit,'-',label='Fit (MCMC)')
    plt.legend(loc=0)
    plt.yscale('log')
    plt.xlabel('Date')
    plt.ylabel(r'$\nu$ ($ms/mHz^{2}$)')
    plt.title('Fit Results')
    pdf.savefig()
    plt.close()