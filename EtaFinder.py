import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import os
import re
import argparse
import dynspec
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(description='Find Eta From Data')
parser.add_argument('-dir', default='./', type=str,help='Data Directory')
parser.add_argument('-ft',default='.npz',type=str,help='File Type')

args=parser.parse_args()

fnames=np.array([list(f for f in os.listdir(args.dir) if f.endswith(args.ft))])[0,:]

if not args.ft[-4:]=='npz':
    for i in range(fnames.shape[0]):
        fname='%s/%s' %(args.dir,fnames[i])
        dynspec.data_to_dspec(fname,profsig=5,sigma=10)

fnames=np.array([list(f for f in os.listdir(args.dir) if f.endswith('npz'))])[0,:]

times=np.zeros(fnames.shape)
for i in range(fnames.shape[0]):
    times[i]=np.load('%s/%s' %(args.dir,fnames[i]))['time'].mean()
    srce=np.load('%s/%s' %(args.dir,fnames[i]))['source']

fnames=fnames[np.argsort(times)]
times=np.sort(times)

eta_est=np.zeros(fnames.shape[0])*u.ms/u.mHz**2
eta_low=np.zeros(fnames.shape[0])*u.ms/u.mHz**2
eta_high=np.zeros(fnames.shape[0])*u.ms/u.mHz**2
with PdfPages('%s/%s_etas.pdf' %(args.dir,srce)) as pdf:
    for i in range(fnames.shape[0]):
        data=np.load('%s/%s' %fnames[i])
        eta_est[i],eta_low[i],eta_high[i]=dynspec.eta_from_data(data['I'],data['freq'],data['time'],rbin=256,xlim=30,ylim=1,tau_lim=.001*u.ms,srce=srce,eta_true=None,prof=data['prof'],template=data['template'])
    plt.figure(figsize=(8,8))
    plt.plot_date(Time(times,format='mjd').plot_date,eta_est,'r',label='Hough')
    plt.plot_date(Time(times,format='mjd').plot_date,eta_est+eta_high,'r')
    plt.plot_date(Time(times,format='mjd').plot_date,eta_est-eta_low,'r')
    # plt.plot_date(Time(times,format='mjd').plot_date,eta_real,'c',label='Real')
    Amps=1/(np.sqrt(2*np.pi)*(eta_high+eta_low))
    mu=Amps*(eta_high**2-eta_low**2)+eta_est
    plt.plot_date(Time(times,format='mjd').plot_date,mu,'k')
    plt.yscale('log')
    plt.legend()
    plt.title(r'$\eta$ Evolution')
    plt.xlabel('Time')
    plt.ylabel(r'$\eta$ ($ms/mHz**2$)')
    pdf.savefig()
    plt.close()
output=np.array([times,eta_est,eta_low,eta_high])
np.save('%s/etas.npy' %args.dir,output)



