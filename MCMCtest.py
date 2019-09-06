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

parser = argparse.ArgumentParser(description='Test Orbit Recovery')
parser.add_argument('-ns',type=int,default=2000,help='Number of Samples')
parser.add_argument('-nw',type=int,default=40,help='Number of Walkers')
parser.add_argument('-oo',type=float,default=0,help='Omega Orbit')
parser.add_argument('-os',type=float,default=0,help='Omega Screen')
parser.add_argument('-i',type=float,default=45,help='Inclination')
parser.add_argument('-s',type=float,default=.5,help='Fractional Screen Distance')
parser.add_argument('-ml',action='store_true',default= False, help='Maximum Likelihood')
parser.add_argument('-mm',type=str,default= 'L-BFGS-B', help='Maximum Likelihood Method')
parser.add_argument('-np',type=int,default= 0, help='Number of Random Parameters')


args=parser.parse_args()
print('Import Complete')

print('Load and Query')
names=list(f[6:-4] for f in os.listdir('./binarytimestamps'))
names_cat=list(names)
names_cat[0]='J1939+2134'
names_cat[6]=names[6]+'+2551'
names.remove(names[6])
names_cat.remove(names_cat[6])
query=QueryATNF(psrs=names_cat)
psrs = query.get_pulsars()

print('Select Source')
Test_source = 'J1643-1224'
srce=SkyCoord.from_name('PSR %s' %Test_source)

print('Define Simulation Parameters')
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
    T0 = Time(brentq(orbfits.T0_find,
                     TASC.mjd, (TASC + Pb).mjd,
                     args=(TASC, -Om_peri, Pb, Ecc)),
              format='mjd')
else:
    T0 = Time(PSR.T0, format='mjd')
pm_ra = (PSR.PMRA * u.mas / u.year).to_value(u.rad / u.s) / u.s
pm_dec = (PSR.PMDec * u.mas / u.year).to_value(u.rad / u.s) / u.s
f0 = 1 * u.GHz

##Unknowns
Om_orb = np.mod(args.oo,360) * u.deg
Om_scr = (np.mod(args.os+90,180)-90) * u.deg
inc = np.mod(args.i,90) * u.deg
ds = dp *args.s

print('Simulate Data')
a = np.abs(A1 / np.sin(inc))
eta_data = orbfits.eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                   dp, ds, f0, pm_ra, pm_dec)

sigma = eta_data / 10

eta_noisy = eta_data + np.random.normal(0, 1, eta_data.shape[0])*sigma

lwrs=np.array([0,-90,0,0])
uprs=np.array([360,90,90,dp.to_value(u.kpc)])



ndim, nwalkers = 4, args.nw
if args.ml:
    nll = lambda *args: -orbfits.lnprob(*args)
    print(nll((lwrs+uprs)/2,eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs))
    result = minimize(nll, (lwrs+uprs)/2, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),bounds=[(lwrs[i],uprs[i]) for i in range(ndim)],method=args.mm)
    print(result.x)
    pos = [result.x + 1e-2*np.random.randn(ndim)*result.x for i in range(nwalkers)]
else:
    pos = [np.random.uniform(lwrs,uprs) for i in range(nwalkers)]


print('Start Walking')
sampler = emcee.EnsembleSampler(nwalkers, ndim, orbfits.lnprob, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),threads=20)

sampler.run_mcmc(pos, args.ns)
print('Walk Complete')

samples = sampler.chain[:, min((1000,args.ns//2)):, :].reshape((-1, ndim))

lwrs=samples.mean(0)-np.array([180,-90,-45,0])
uprs=samples.mean(0)+np.array([360,180,90,0])
lwrs[-1]=0
uprs[-1]=dp.to_value(u.kpc)
pos=sampler.chain[:,-1,:]
print('Start Walk 2')
sampler.run_mcmc(pos, args.ns)
samples = sampler.chain[:, min((1000,args.ns//2)):, :].reshape((-1, ndim))
print('Walk Complete')


para_names=np.array([r'$\Omega_{orb}$',r'$\Omega_{scr}$',r'$i$',r'$D_s$'])
para_names_file=np.array(['OmOrb','OmScr','i','Ds'])
reals=np.array([Om_orb.value,Om_scr.value,inc.value,ds.value])

fig = corner.corner(samples, labels=para_names,
                      truths=reals)
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
plt.plot_date(times_curve.plot_date,eta_fit,'-',label='Fit')
plt.plot_date(times_curve.plot_date,eta_real,'-',label='Real')
plt.legend(loc=0)
plt.yscale('log')
plt.xlabel('Date')
plt.ylabel(r'$\nu$ ($ms/mHz^{2}$)')
plt.title('Fit Results')
plt.savefig('FIT.png')

np.savez('samples.npz',samps=sampler.chain,reals=reals)

for i in range(args.np):
    lwrs=np.array([0,-90,0,0])
    uprs=np.array([360,90,90,dp.to_value(u.kpc)])
    ##Unknowns
    Om_orb,Om_scr,inc,ds=np.random.unform(lwrs,uprs)

    print('Simulate Data')
    a = np.abs(A1 / np.sin(inc))
    eta_data = orbfits.eta_orb(srce,times,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                    dp, ds, f0, pm_ra, pm_dec)

    sigma = eta_data / 10

    eta_noisy = eta_data + np.random.normal(0, 1, eta_data.shape[0])*sigma

    if args.ml:
        nll = lambda *args: -orbfits.lnprob(*args)
        print(nll((lwrs+uprs)/2,eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs))
        result = minimize(nll, (lwrs+uprs)/2, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),bounds=[(lwrs[i],uprs[i]) for i in range(ndim)],method=args.mm)
        print(result.x)
        pos = [result.x + 1e-2*np.random.randn(ndim)*result.x for i in range(nwalkers)]
    else:
        pos = [np.random.uniform(lwrs,uprs) for i in range(nwalkers)]


    print('Start Walking')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, orbfits.lnprob, args=(eta_noisy,sigma,srce,times,Ecc,T0, Pb, Om_peri_dot, Om_peri,dp,f0,pm_ra,pm_dec, A1,lwrs,uprs),threads=20)

    sampler.run_mcmc(pos, args.ns)
    print('Walk Complete')

    samples = sampler.chain[:, min((1000,args.ns//2)):, :].reshape((-1, ndim))

    lwrs=samples.mean(0)-np.array([180,-90,-45,0])
    uprs=samples.mean(0)+np.array([360,180,90,0])
    lwrs[-2:]=np.array([0,0])
    uprs[-2:]=np.array([90,dp.to_value(u.kpc)])
    pos=sampler.chain[:,-1,:]
    print('Start Walk 2')
    sampler.run_mcmc(pos, args.ns)
    samples = sampler.chain[:, min((1000,args.ns//2)):, :].reshape((-1, ndim))
    print('Walk Complete')


    para_names=np.array([r'$\Omega_{orb}$',r'$\Omega_{scr}$',r'$i$',r'$D_s$'])
    para_names_file=np.array(['OmOrb','OmScr','i','Ds'])
    reals=np.array([Om_orb.value,Om_scr.value,inc.value,ds.value])

    fig = corner.corner(samples, labels=para_names,
                        truths=reals)
    fig.savefig("Corner_%s.png" %(i+1))

    for k in range(4):
        plt.figure()
        for i in range(nwalkers):
            plt.plot(sampler.chain[i,:,k])
        plt.title(para_names[k])
        plt.axhline(reals[k],color='k',linewidth=2)
        plt.savefig('%s_walk_%s.png' %(para_names_file[k],i+1))

    times_curve=Time(np.linspace(times.min().mjd,times.max().mjd,10000),format='mjd')

    eta_fit = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, samples[:,0].mean()*u.deg, samples[:,1].mean()*u.deg, samples[:,2].mean()*u.deg,
                    dp, samples[:,3].mean()*u.kpc, f0, pm_ra, pm_dec)
    eta_real = orbfits.eta_orb(srce,times_curve,Ecc, a, T0, Pb, Om_peri_dot, Om_peri, Om_orb, Om_scr, inc,
                    dp, ds, f0, pm_ra, pm_dec)

    plt.figure()
    plt.plot_date(times.plot_date,eta_noisy,label='Data')
    plt.plot_date(times_curve.plot_date,eta_fit,'-',label='Fit')
    plt.plot_date(times_curve.plot_date,eta_real,'-',label='Real')
    plt.legend(loc=0)
    plt.yscale('log')
    plt.xlabel('Date')
    plt.ylabel(r'$\nu$ ($ms/mHz^{2}$)')
    plt.title('Fit Results')
    plt.savefig('FIT.png')

    np.savez('samples_%s.npz' %(i+1),samps=sampler.chain,reals=reals)