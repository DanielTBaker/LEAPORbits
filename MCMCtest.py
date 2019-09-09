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
from emcee.utils import MPIPool
import itertools

parser = argparse.ArgumentParser(description='Test Orbit Recovery')
parser.add_argument('-ns',type=int,default=2000,help='Number of Samples')
parser.add_argument('-nw',type=int,default=80,help='Number of Walkers')
parser.add_argument('-oo',type=float,default=0,help='Omega Orbit')
parser.add_argument('-os',type=float,default=0,help='Omega Screen')
parser.add_argument('-i',type=float,default=45,help='Inclination')
parser.add_argument('-s',type=float,default=.5,help='Fractional Screen Distance')
parser.add_argument('-ml',action='store_true',default= False, help='Maximum Likelihood')
parser.add_argument('-es',action='store_true',default= False, help='Equally Spaced')
parser.add_argument('-mm',type=str,default= 'L-BFGS-B', help='Maximum Likelihood Method')
parser.add_argument('-np',type=int,default= 0, help='Number of Random Parameters')
parser.add_argument('-nt',type=int,default=80,help='Number of Threads')


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



ndim, nwalkers, nthreads = 4, args.nw, args.nt
if args.ml:
    nll = lambda *args: -orbfits.lnprob(*args)
    print('Maximum Likelihood',flush=True)
    initial=(lwrs+uprs)/2
    initial[1]=(np.mod(np.arctan2(dp*pm_ra,dp*pm_dec).to_value(u.deg),180)-90)
    result = minimize(nll, initial, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),bounds=[(lwrs[i],uprs[i]) for i in range(ndim)],method=args.mm)
    pos = np.random.normal(1,1e-2,(nwalkers,ndim))*result.x[np.newaxis,:]
elif args.es:
    p1=np.array([60,120,180,240,300])
    p2=np.array([-54,-18,18,54])
    p3=np.array([33,66])
    p4=np.array([.2,.4,.6,.8])*dp.to_value(u.kpc)
    pos=np.array([(p1[i],p2[j],p3[k],p4[l]) for i,j,k,l in itertools.product(range(len(p1)),range(len(p2)),range(len(p3)),range(len(p4)))])
else:
    pos = np.random.uniform(0,1,(nwalkers,ndim))*(uprs-lwrs)[np.newaxis,:]+lwrs[np.newaxis,:]

def lnp(theta):
    return(0)
print('Start Walking',flush=True)
sampler = emcee.EnsembleSampler(nwalkers, ndim, orbfits.lnprob, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),threads=nthreads)

sampler.run_mcmc(pos, args.ns)

samples = sampler.chain[:, min((1000,args.ns//2)):, :].reshape((-1, ndim))

para_names=np.array([r'$\Omega_{orb}$',r'$\Omega_{scr}$',r'$i$',r'$D_s$'])
para_names_file=np.array(['OmOrb','OmScr','i','Ds'])
reals=np.array([Om_orb.value,Om_scr.value,inc.value,ds.value])

fig = corner.corner(samples, labels=para_names,
                      truths=reals,quantiles=[.16, .50, .84],show_titles=True)
fig.savefig("Corner.png")

for k in range(4):
    plt.figure()
    for i in range(nwalkers):
        plt.plot(sampler.chain[i,:,k])
    plt.title(para_names[k])
    plt.axhline(reals[k],color='k',linewidth=2)
    plt.savefig('%s_walk.png' %para_names_file[k])

times_curve=Time(np.linspace(times.min().mjd,times.max().mjd,10000),format='mjd')

eta_fit = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, samples[:,0].mean()*u.deg, samples[:,1].mean()*u.deg, samples[:,2].mean()*u.deg,
                   dp, samples[:,3].mean()*u.kpc, f0, pm_ra, pm_dec)
eta_real = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                   dp, ds, f0, pm_ra, pm_dec)

plt.figure()
plt.plot_date(times.plot_date,eta_noisy,label='Data')
plt.plot_date(times_curve.plot_date,eta_real,'-',label='Real')
plt.plot_date(times_curve.plot_date,eta_fit,'-',label='Fit (MCMC)')
if args.ml:
    eta_ML = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, result.x[0]*u.deg, result.x[1]*u.deg, result.x[2]*u.deg,
                dp, result.x[3]*u.kpc, f0, pm_ra, pm_dec)
    plt.plot_date(times_curve.plot_date,eta_ML,'-',label='Fit (ML)')
plt.legend(loc=0)
plt.yscale('log')
plt.xlabel('Date')
plt.ylabel(r'$\nu$ ($ms/mHz^{2}$)')
plt.title('Fit Results')
plt.savefig('FIT.png')

np.savez('samples.npz',samps=sampler.chain,reals=reals)

for param_num in range(args.np):
    lwrs=np.array([0,-90,0,0])
    uprs=np.array([360,90,90,dp.to_value(u.kpc)])
    ##Unknowns
    Om_orb,Om_scr,inc,ds=np.random.uniform(lwrs,uprs)
    Om_orb*=u.deg
    Om_scr*=u.deg
    inc*=u.deg
    ds*=u.kpc
    print('\033[4ARandom Parameters %s' %(param_num+1),flush=True)
    print('Simulate Data',flush=True)
    a = np.abs(A1 / np.sin(inc))
    eta_data = orbfits.eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                    dp, ds, f0, pm_ra, pm_dec)

    sigma = eta_data / 10

    eta_noisy = eta_data + np.random.normal(0, 1, eta_data.shape[0])*sigma

    if args.ml:
        nll = lambda *args: -orbfits.lnprob(*args)
        print('Maximum Likelihood',flush=True)
        initial=(lwrs+uprs)/2
        initial[1]=(np.mod(np.arctan2(dp*pm_ra,dp*pm_dec).to_value(u.deg),180)-90)
        result = minimize(nll, initial, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),bounds=[(lwrs[i],uprs[i]) for i in range(ndim)],method=args.mm)
        pos = np.random.normal(1,1e-2,(nwalkers,ndim))*result.x[np.newaxis,:]
    elif args.es:
        p1=np.array([60,120,180,240,300])
        p2=np.array([-54,-18,18,54])
        p3=np.array([33,66])
        p4=np.array([.2,.4,.6,.8])*dp.to_value(u.kpc)
        pos=np.array([(p1[i],p2[j],p3[k],p4[l]) for i,j,k,l in itertools.product(range(len(p1)),range(len(p2)),range(len(p3)),range(len(p4)))])
    else:
        pos = np.random.uniform(0,1,(nwalkers,ndim))*(uprs-lwrs)[np.newaxis,:]+lwrs[np.newaxis,:]


    print('Start Walking',flush=True)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, orbfits.lnprob, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),threads=nthreads)

    sampler.run_mcmc(pos, args.ns)

    samples = sampler.chain[:, min((1000,args.ns//2)):, :].reshape((-1, ndim))

    para_names=np.array([r'$\Omega_{orb}$',r'$\Omega_{scr}$',r'$i$',r'$D_s$'])
    para_names_file=np.array(['OmOrb','OmScr','i','Ds'])
    reals=np.array([Om_orb.value,Om_scr.value,inc.value,ds.value])

    fig = corner.corner(samples, labels=para_names,
                        truths=reals,quantiles=[.16, .50, .84],show_titles=True)
    fig.savefig("Corner_%s.png" %(param_num+1))

    for k in range(4):
        plt.figure()
        for i in range(nwalkers):
            plt.plot(sampler.chain[i,:,k])
        plt.title(para_names[k])
        plt.axhline(reals[k],color='k',linewidth=2)
        plt.savefig('%s_walk_%s.png' %(para_names_file[k],param_num+1))

    times_curve=Time(np.linspace(times.min().mjd,times.max().mjd,10000),format='mjd')

    eta_fit = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, samples[:,0].mean()*u.deg, samples[:,1].mean()*u.deg, samples[:,2].mean()*u.deg,
                    dp, samples[:,3].mean()*u.kpc, f0, pm_ra, pm_dec)
    eta_real = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                    dp, ds, f0, pm_ra, pm_dec)


    plt.figure()
    plt.plot_date(times.plot_date,eta_noisy,label='Data')
    plt.plot_date(times_curve.plot_date,eta_real,'-',label='Real')
    plt.plot_date(times_curve.plot_date,eta_fit,'-',label='Fit (MCMC)')
    if args.ml:
        eta_ML = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, result.x[0]*u.deg, result.x[1]*u.deg, result.x[2]*u.deg,
                    dp, result.x[3]*u.kpc, f0, pm_ra, pm_dec)
        plt.plot_date(times_curve.plot_date,eta_ML,'-',label='Fit (ML)')
    plt.legend(loc=0)
    plt.yscale('log')
    plt.xlabel('Date')
    plt.ylabel(r'$\nu$ ($ms/mHz^{2}$)')
    plt.title('Fit Results')
    plt.savefig('FIT_%s.png' %(param_num+1))

    np.savez('samples_%s.npz' %(param_num+1),samps=sampler.chain,reals=reals)
