import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import os
import re
import argparse
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import ththmod

parser = argparse.ArgumentParser(description='Find Eta From Data')
parser.add_argument('-dir', default='', type=str,help='Data Directory')
parser.add_argument('-fmax',default=np.inf,type=float,help='Maximum observation frequency')
parser.add_argument('-tmin',default=1,type=float,help='Minimum Number of Time Steps')



args=parser.parse_args()
dirname=args.dir
fmax=args.fmax
tmin=args.tmin

fnames=np.array([list(f for f in os.listdir(dirname) if f.endswith('npz'))])[0,:]

times=np.zeros(fnames.shape)
fpeak=np.zeros(fnames.shape)
f0=np.zeros(fnames.shape)
tl=np.zeros(times.shape)
for i in range(fnames.shape[0]):
    files=np.load('%s/%s' %(dirname,fnames[i]))
    times[i]=files['time'].mean()
    srce=files['source']
    f0[i]=files['freq'].mean()
    fpeak[i]=files['freq'].max()
    tl[i]=files['time'].shape[0]

fnames=fnames[fpeak<fmax]
times=times[fpeak<fmax]
f0=f0[fpeak<fmax]
tl=tl[fpeak<fmax]

fnames=fnames[tl>=tmin]
times=times[tl>=tmin]
f0=f0[tl>=tmin]

fnames=fnames[np.argsort(times)]
times=np.sort(times)

fnames=fnames[:4]

for i in range(fnames.shape[0]):
    files=np.load('%s/%s' %(dirname,fnames[i]))
    freq=files['freq']*u.MHz
    time=(files['time']*u.day).to(u.s)
    time-=time[0]
    dspec=files['I'].T
    dspec=np.reshape(dspec,(dspec.shape[0]//4,4,-1)).mean(1)
    dspec/=svd_model(dspec).real
    dspec_pad=np.pad(dspec,((0,dspec.shape[0]),(0,dspec.shape[1])),mode='constant',constant_values=dspec.mean())
    SS=np.fft.fft2(dspec_pad)
    fd=np.fft.fftfreq(dspec_pad.shape[1],time[1]).to(u.mHz)
    tau=np.fft.fftfreq(dspec_pad.shape[0],freq[1]-freq[0]).to(u.us)

    msk=(tau[:,np.newaxis]>.05*u.us) * (np.abs(fd[np.newaxis,:])>.5*u.mHz)
    eta_low=.4*u.us/((3*u.mHz)**2)
    eta_high=tau.max()/((1*u.mHz)**2)
    etas_abs,chisq_abs,etas,chisq,eta_fit,eta_sig=ththmod.eta_full(SS,
                                                            fd,
                                                            tau,
                                                            msk,
                                                            np.abs(SS)**2,
                                                            fd,
                                                            tau,
                                                            msk,
                                                            5,
                                                            eta_low,
                                                            eta_high,
                                                            nth1=1024,
                                                            nth2=1024,
                                                            neta1=10,
                                                            neta2=10,
                                                            thth_fit=False)
    
    SS_min=(np.abs(SS[np.abs(tau)>5*tau.max()/6,:][:,np.abs(fd)>5*fd.max()/6])**2).mean()*10
    SS_max=(np.abs(SS[1:,1:])**2).max()/10
    SS_ext=[(fd.min()-fd[1]/2).value,
            (fd.max()+fd[1]/2).value,
            (tau.min()-tau[1]/2).value,
            (tau.max()+tau[1]/2).value]
    dspec_ext=[0,time[-1].to_value(u.min),freq[0].value,freq[-1].value]
    plt.figure()
    plt.imshow(np.fft.fftshift(np.abs(SS)**2),
                origin='lower',
                aspect='auto',
                norm=LogNorm(),
                extent=SS_ext,
                vmin=SS_min,
                vmax=SS_max)
    plt.xlim((-5,5))
    plt.ylim((0,.5))
    plt.figure()
    plt.imshow(dspec,
                origin='lower',
                aspect='auto',
                extent=dspec_ext)