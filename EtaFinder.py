import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import os
import re
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from scintillation.sstools import slowft as slowft
from scintillation.sstools import ss_tools as sstools
import datareader

parser = argparse.ArgumentParser(description='Find Eta From Data')
parser.add_argument('-dir', default='', type=str,help='Data Directory')
parser.add_argument('-ft',default='.npz',type=str,help='File Type')

args=parser.parse_args()

dirname=args.dir
ftype=args.ft

fnames=np.array([list(f for f in os.listdir(dirname) if f.endswith(ftype))])[0,:]

dirname_save=dirname
if not args.ft[-4:]=='npz':
    for i in range(fnames.shape[0]):
        fname='%s/%s' %(dirname,fnames[i])
        times, freqs,N,dynspec,temp0,template,srce=datareader.data_to_dspec(fname,profsig=5,sigma=10)
        fname=fnames[i]
        while not fname.endswith('.'):
            fname=fname[:-1]
        try:
            np.savez('%s/%snpz' %(dirname,fname),I=dynspec,freq=freqs,time=times,N=N,prof=temp0,template=template,source=source)
        except:
            np.savez('%snpz' %(fname),I=dynspec,freq=freqs,time=times,N=N,prof=temp0,template=template,source=source)
            dirname_save='./'

fnames=np.array([list(f for f in os.listdir(dirname_save) if f.endswith('npz'))])[0,:]

times=np.zeros(fnames.shape)
for i in range(fnames.shape[0]):
    times[i]=np.load('%s/%s' %(dirname_save,fnames[i]))['time'].mean()
    srce=np.load('%s/%s' %(dirname_save,fnames[i]))['source']

fnames=fnames[np.argsort(times)]
times=np.sort(times)

eta_est=np.zeros(fnames.shape[0])*u.ms/u.mHz**2
eta_low=np.zeros(fnames.shape[0])*u.ms/u.mHz**2
eta_high=np.zeros(fnames.shape[0])*u.ms/u.mHz**2
# with PdfPages('%s/%s_etas.pdf' %(dirname_save,srce)) as pdf:
#     for i in range(fnames.shape[0]):
#         data=np.load('%s/%s' %(dirname_save,fnames[i]))
#         eta_est[i],eta_low[i],eta_high[i]=datareader.eta_from_data(data['I'],data['freq'],data['time'],rbin=256,xlim=30,ylim=1,tau_lim=.001*u.ms,srce=srce,eta_true=None,prof=data['prof'],template=data['template'])
#     plt.figure(figsize=(8,8))
#     plt.plot_date(Time(times,format='mjd').plot_date,eta_est,'r',label='Hough')
#     plt.plot_date(Time(times,format='mjd').plot_date,eta_est+eta_high,'r')
#     plt.plot_date(Time(times,format='mjd').plot_date,eta_est-eta_low,'r')
#     # plt.plot_date(Time(times,format='mjd').plot_date,eta_real,'c',label='Real')
#     Amps=1/(np.sqrt(2*np.pi)*(eta_high+eta_low))
#     mu=Amps*(eta_high**2-eta_low**2)+eta_est
#     plt.plot_date(Time(times,format='mjd').plot_date,mu,'k')
#     plt.yscale('log')
#     plt.legend()
#     plt.title(r'$\eta$ Evolution')
#     plt.xlabel('Time')
#     plt.ylabel(r'$\eta$ ($ms/mHz**2$)')
#     pdf.savefig()
#     plt.close()

with PdfPages('%s/%s_etas.pdf' %(dirname_save,srce)) as pdf:
    for i in range(times.shape[0]):
        print('Start %s / %s' %(i+1,times.shape[0]))
        data=np.load('%s/%s' %(dirname_save,fnames[i]))
        eta_est[i],eta_low[i],eta_high[i]=datareader.eta_from_data(data['I'],data['freq'],data['time'],rbin=256,xlim=30,ylim=1,tau_lim=0*u.ms,srce=srce,prof=data['prof'],template=data['template'])
        pdf.savefig()
    plt.figure(figsize=(8,8))
    ymin=.9*min(((eta_est-eta_low).min().value))
    ymax=1.5*max(((eta_est[eta_high<np.inf]+eta_high[eta_high<np.inf]).max().value))
    ebars=np.array([eta_low.value,eta_high.value])
    ebars[ebars==np.inf]=10*ymax
    plt.errorbar(Time(times,format='mjd').plot_date,eta_est.value,ebars,label='Measured')
    plt.ylim((ymin,ymax))
    plt.yscale('log')
    plt.legend()
    plt.title(r'$\eta$ Evolution')
    plt.xlabel('Time')
    plt.ylabel(r'$\eta$ ($ms/mHz**2$)')
    pdf.savefig()
output=np.array([times,eta_est,eta_low,eta_high])
np.save('%s/eta_params.npy' %dirname_save,output)




