import matplotlib as mpl
mpl.use('Agg')
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
parser.add_argument('-dirC', default='', type=str,help='Calibration Directory')
parser.add_argument('-dirS',default='', type=str,help='Save Directory')
parser.add_argument('-ft',default='.npz',type=str,help='File Type') 
parser.add_argument('-fP', default=10,type=float,help='f_D limit for plots')
parser.add_argument('-tP', default=1,type=float,help='tau limit for plots')
parser.add_argument('-tH', default=.05,type=float,help='Lowest tau for Hough Transform (us)')
parser.add_argument('-fH', default=.05,type=float,help='Lowest f_D for Hough Transform (mHz)')
parser.add_argument('-rbin', default=0,type=int,help='Number of bins in rebinned SS')
parser.add_argument('-rbd', default=1,type=int,help='Rebinning factor in DS')
parser.add_argument('-Nr',default=5,type=float,help='S/N ratio for Hough Cutoff')



args=parser.parse_args()

dirname=args.dir
cal_dirname=args.dirC
dirname_save=args.dirS
ftype=args.ft


fnames=np.array([list(f for f in os.listdir(dirname) if f.endswith(ftype))])[0,:]

if not args.ft[-4:]=='npz':
    fnames_cals=np.array([list(f for f in os.listdir(cal_dirname) if f.endswith('cf'))])[0,:]
    t_cals=np.zeros(fnames_cals.shape[0])
    for i in range(fnames_cals.shape[0]):
        t_cals[i]=datareader.cal_time('%s/%s' %(cal_dirname,fnames_cals[i]))
    for i in range(fnames.shape[0]):
        if not os.path.exists('%s/%snpz' %(dirname_save,%fnames[i][:-len(ftype)]))
            try:
                print('Starting: %s' %fnames[i])
                fname='%s/%s' %(dirname,fnames[i])
                t_last_cal=datareader.cal_find(fname,t_cals)
                fname_cal='%s/%s' %(cal_dirname,fnames_cals[t_cals==t_last_cal][0])
                times, freqs,N,dynspec,temp0,template,srce=datareader.data_to_dspec(fname,fname_cal,profsig=5,sigma=10)
                fname=fnames[i]
                while not fname.endswith('.'):
                    fname=fname[:-1]
                np.savez('%s/%snpz' %(dirname_save,fname),I=dynspec,freq=freqs,time=times,N=N,prof=temp0,template=template,source=srce)
            except:
                print('Check Calibration')
        else:
            print('File Already Exists')
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
use_data=np.ones(fnames.shape[0]).astype(bool)
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
        try:
            print('Start %s / %s' %(i+1,times.shape[0]))
            data=np.load('%s/%s' %(dirname_save,fnames[i]))
            if args.rbin==0:
                rbin=data['freq'].shape[0]
            else:
                rbin=args.rbin
            dspec=data['I']
            eta_est[i],eta_low[i],eta_high[i]=datareader.eta_from_data(dspec,data['freq'],data['time'],rbin=rbin,rbd=args.rbd,xlim=args.fP,ylim=args.tP,tau_lim=args.tH*u.us,fd_lim=args.fH*u.mHz,srce=srce,prof=data['prof'],template=data['template'],Nr=args.Nr)
            pdf.savefig()
        except:
            use_data[i]=False
    eta_est=eta_est[use_data]
    eta_high=eta_high[use_data]
    eta_low=eta_low[use_data]
    times=times[use_data]
    plt.figure(figsize=(8,8))
    ymin=.9*(eta_est-eta_low).min().value
    ymax=1.5*(eta_est[eta_high<np.inf]+eta_high[eta_high<np.inf]).max().value
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




