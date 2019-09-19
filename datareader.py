import psrchive
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from scintillation.sstools import slowft as slowft
from scintillation.sstools import ss_tools as sstools
from scipy.ndimage.filters import gaussian_filter
import scipy
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
import scipy.signal

def data_to_dspec(fname,profsig=5,sigma=10):
    arch = psrchive.Archive_load(fname)
    arch.dedisperse()
    data = arch.get_data()
    dc = data[:,(0,1)].mean(1)

    # Create folded spectrum, after offset and scaling
    foldspec, mask = sstools.clean_foldspec(dc, plots=False)

    # Get metadata
    source = arch.get_source()
    a = arch.start_time()
    t0 = a.strtempo()
    t0 = Time(float(t0), format='mjd', precision=0)

    # Create profile by summing over time, frequency, normalize peak to 1
    template = foldspec.mean(0).mean(0)
    template /= np.max(template)
    temp0=np.copy(template)

    # Noise from bottom 50% of profile
    tnoise = np.std(template[template<np.median(template)])
    template[template < tnoise*profsig] = 0

    # Multiply the profile by the template, sum over phase
    dynspec = (foldspec*template[np.newaxis,np.newaxis,:]).mean(-1)
    dynspec /= np.std(dynspec)
    dynspec[dynspec > sigma] = np.mean(dynspec)
    
    
    dspec=np.copy(dynspec)
    #if taper (Use Tukey window to taper edges of dynspec)
    t_window = scipy.signal.windows.tukey(dynspec.shape[0], alpha=0.2, sym=True)
    dspec *= t_window[:,np.newaxis]
    f_window = scipy.signal.windows.tukey(dynspec.shape[1], alpha=0.2, sym=True)
    dspec *= f_window[np.newaxis,:]

    #if pad (ADD PADDING, MASK REMOVAL)
    nt = dynspec.shape[0]
    nf = dynspec.shape[1]
    dspec = np.pad(dspec,((0,nt),(0,nf)),mode='constant',constant_values=0)
    SS = np.fft.fft2(dspec)/np.sqrt(nf*nt)

    # Get frequency and time info for plot axes
    try:
        freqs = arch.get_frequencies()
    except:
        midf = arch.get_centre_frequency()
        bw = arch.get_bandwidth()
        freqs = np.linspace(midf-bw/2., midf+bw/2., nf, endpoint=False)
    nt = dynspec.shape[0]
    T = arch.integration_length()

    # 2D power spectrum is the Secondary spectrum
    SS = np.fft.fftshift(SS)
    SS = abs(SS)**2.0
    fd = np.fft.fftshift(np.fft.fftfreq(dspec.shape[1]))
    tau = np.fft.fftshift(np.fft.fftfreq(dspec.shape[0]))
    N = SS[np.abs(tau)>9*tau.max()/10,:][:,np.abs(fd)>9*fd.max()/10].mean()
    
    times=np.linspace(t0.mjd,(t0+T*u.s).mjd,nt)
    return(times, freqs,N,dynspec,temp0,template,source)

# def Hough(C,N,tau,fd,tau_lim,normed=False):
#     Vals=C[C>2*N]
#     x=fd[np.newaxis,:]*np.ones(C.shape)
#     y=tau[:,np.newaxis]*np.ones(C.shape)
#     x=x[C>2*N]
#     y=y[C>2*N]
#     x=x[y>tau_lim]
#     Vals=Vals[y>tau_lim]
#     y=y[y>tau_lim]
#     eta_low=(y-(tau[1]-tau[0])/2)/(np.abs(x)+(fd[1]-fd[0])/2)**2
#     eta_high=(y+(tau[1]-tau[0])/2)/(np.abs(x)-(fd[1]-fd[0])/2)**2
#     eta_high[np.abs(x)<=(fd[1]-fd[0])/2]=(y[np.abs(x)<=(fd[1]-fd[0])/2]+(tau[1]-tau[0])/2)/((fd[1]-fd[0])/100)**2
#     etas=np.concatenate(((eta_low.value+eta_high.value)/2,np.array([eta_low.min().value,eta_high.max().value])))
#     etas=np.sort(etas)*u.ms/u.mHz**2
#     HT=np.zeros(etas.shape)
#     if normed:
#         Vals/=(eta_high-eta_low).value
#     for i in range(eta_low.shape[0]):
#         HT[etas>eta_low[i]]+=Vals[i]
#         HT[etas>eta_high[i]]-=Vals[i]
#     return(etas,HT)


# def Hough_Rob(SS,tau,fd,rbin,tau_lim):
#     bintau = int(SS.shape[0] // rbin)
#     SSb = SS.reshape(SS.shape[0]//bintau, bintau,-1).mean(1)
#     tau*=u.us
#     tau=tau.to(u.ms)
#     fd*=u.mHz

#     SSb2=np.fft.ifftshift(SSb)
#     tau2=np.fft.ifftshift(tau)[::bintau]*u.ms
#     fd2=np.fft.ifftshift(fd)*u.mHz
#     N=SSb2[np.abs(tau2)>4*tau2.max()/5,:][:,np.abs(fd2)>4*fd2.max()/5].mean()
#     etas,HT=Hough(SSb2,N,tau2,fd2,tau_lim)
#     fd=fd.value
#     tau=tau.to_value(u.us)
#     return(etas,HT)

def Hough_Prob(C,N,tau,fd,tau_lim,normed=False):
    Vals=C[C>5*N]
    x=fd[np.newaxis,:]*np.ones(C.shape)
    y=tau[:,np.newaxis]*np.ones(C.shape)
    x=x[C>5*N]
    y=y[C>5*N]
#     y=y[np.abs(x)>(fd[1]-fd[0])/2]
#     Vals=Vals[np.abs(x)>(fd[1]-fd[0])/2]
#     x=x[np.abs(x)>(fd[1]-fd[0])/2]
    x=x[y>tau_lim]
    Vals=Vals[y>tau_lim]
    y=y[y>tau_lim]
    print(Vals.shape)
    
    dfd=(fd[1]-fd[0])/2
    dtau=(tau[1]-tau[0])/2
    
    xc=x[:,np.newaxis]+np.array([1.,1.,-1.,-1.])[np.newaxis,:]*dfd
    yc=y[:,np.newaxis]+np.array([1.,-1.,1.,-1.])[np.newaxis,:]*dtau
    
    eta_low=(y-dtau)/(np.abs(x)+dfd)**2
    eta_high=(y+dtau)/(np.abs(x)-dfd)**2
    eta_high[np.abs(x)<=dfd]=(y[np.abs(x)<=dfd]+dtau)/(dfd/50)**2
    etas=np.concatenate(((eta_low.value+eta_high.value)/2,np.array([eta_low.min().value,eta_high.max().value])))
    etas=np.sort(etas)*u.ms/u.mHz**2
    HT=np.zeros(etas.shape)
    sig_low=np.zeros(HT.shape)*u.ms/u.mHz**2
    sig_high=np.zeros(HT.shape)*u.ms/u.mHz**2
    PN=(-Vals/N)-np.log(N)
    if normed:
        Vals/=(eta_high-eta_low).value
    for i in range(etas.shape[0]):
        uA = (eta_low<=etas[i])*(eta_high>=etas[i])
        xcA = xc[uA,:]
        ycA = yc[uA,:]
        below = yc[:,:,np.newaxis,np.newaxis] < ycA[np.newaxis,np.newaxis,:,:] - etas[i]*(xc[:,:,np.newaxis,np.newaxis]-xcA[np.newaxis,np.newaxis,:,:])**2
        any_below = np.any(below,axis=(1,2,3))
        any_above = np.any(np.invert(below),axis=(1,2,3))
        uAl = any_above*any_below +uA
#         val=np.sum(PN[np.invert(uAl)])
#         HT[i]=np.exp(val)
        val=np.sum(Vals[uAl])
        HT[i]=val
        x_max=np.abs(x[uA]).max()
        y_max=y[uA][np.abs(x[uA])==x_max].max()
        if x_max<=dfd:
            sig_high[i]=np.inf
            sig_low[i]=(y_max-dtau)/(x_max+dfd)**2
        else:
            sig_high[i]=(y_max+dtau)/(x_max-dfd)**2
            sig_low[i]=(y_max-dtau)/(x_max+dfd)**2
        if np.mod(i+1,etas.shape[0]//10)==0:
            print('%s/%s (%s)' %(i+1,etas.shape[0],val))
        
    return(etas,HT,sig_low,sig_high)

def Hough_Rob(SS,tau,fd,rbin,tau_lim):
    bintau = int(SS.shape[0] // rbin)
    SSb = SS.reshape(SS.shape[0]//bintau, bintau,-1).mean(1)
    tau*=u.us
    tau=tau.to(u.ms)
    fd*=u.mHz

    SSb2=np.fft.ifftshift(SSb)
    tau2=np.fft.ifftshift(tau)[::bintau]*u.ms
    fd2=np.fft.ifftshift(fd)*u.mHz
    N=SSb2[np.abs(tau2)>4*tau2.max()/5,:][:,np.abs(fd2)>4*fd2.max()/5].mean()
    etas,HT,sig_low,sig_high=Hough_Prob(SSb2,N,tau2,fd2,tau_lim)
    fd=fd.value
    tau=tau.to_value(u.us)
    return(etas,HT,sig_low,sig_high)

def half_gau(theta,eta):
    eta_est,Amp,eta_low,eta_high=theta
    res=Amp*np.exp(-np.power((eta-eta_est)/eta_low,2)/2)
    res[eta>eta_est]=Amp*Amp*np.exp(-np.power((eta[eta>eta_est]-eta_est)/eta_high,2)/2)
    return(res)

def hg_fit(theta,eta,data):
    model=half_gau(theta,eta)
    return((model-data)**2)

# def eta_from_data(dynspec,freqs,times,rbin=1,xlim=30,ylim=1,tau_lim=.001*u.ms,srce='',eta_true=None,prof=np.ones(10),template=np.ones(10)):
#     nf=dynspec.shape[1]
#     nt=dynspec.shape[0]
#     df = (freqs[1]-freqs[0])*u.MHz
#     dt = (((times[-1]-times[0]) / nt)*u.day).to(u.s)
#     T=(dt*nt).value
    
#     dspec=np.copy(dynspec)
#     #if taper (Use Tukey window to taper edges of dynspec)
#     t_window = scipy.signal.windows.tukey(dynspec.shape[0], alpha=0.2, sym=True)
#     dspec *= t_window[:,np.newaxis]
#     f_window = scipy.signal.windows.tukey(dynspec.shape[1], alpha=0.2, sym=True)
#     dspec *= f_window[np.newaxis,:]

#     #if pad (ADD PADDING, MASK REMOVAL)
#     dspec = np.pad(dspec,((0,nt),(0,nf)),mode='constant',constant_values=0)
#     SS = np.fft.fft2(dspec)/np.sqrt(nf*nt)
#     SS = np.fft.fftshift(SS)
#     SS = abs(SS)**2.0
    
#     bintau = int(SS.shape[1] // rbin)

#     SSb = SS.reshape(-1,SS.shape[1]//bintau, bintau).mean(-1)

#     # Calculate the confugate frequencies (time delay, fringe rate), only used for plotting
#     ft = np.fft.fftfreq(SS.shape[0], dt)
#     ft = np.fft.fftshift(ft.to(u.mHz).value)

#     tau = np.fft.fftfreq(SS.shape[1], df)
#     tau = np.fft.fftshift(tau.to(u.microsecond).value)

#     slow = np.median(SSb)*10**(-0.3)
#     shigh = np.max(SSb)*10**(-1.5)

#     # Hough Transform
#     etas,HT=Hough_Rob(SS.T,tau,ft,rbin,tau_lim)
#     eta_est=etas[HT==HT.max()][0]
#     eta_low=etas[HT>HT.max()*np.exp(-1./2)].min()
#     eta_high=etas[HT>HT.max()*np.exp(-1./2)].max()
#     bounds=((0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf))
#     res=minimize(hg_fit,np.array([eta_est.value,1,eta_low.value,eta_high.value]),args=(etas[etas<eta_high],HT[etas<eta_high]/HT.max()),bounds=bounds)

#     eta_est,Amp,eta_low,eta_high=res.x
#     eta_est*=u.ms/u.mHz**2
#     eta_high*=u.ms/u.mHz**2
#     eta_low*=u.ms/u.mHz**2
    
#     nbin = dynspec.shape[0]//2
#     dspec_plot = dynspec[:nbin*2].reshape(nbin, 2, dynspec.shape[-1]).mean(1)
#     plt.figure(figsize=(10,10))
#     ax1 = plt.subplot2grid((3, 2), (0, 0))
#     ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
#     ax3 = plt.subplot2grid((3, 2), (1, 0))
#     ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

#     plt.subplots_adjust(wspace=0.1)

#     profplot=np.concatenate((prof,prof))
#     profplot2=np.concatenate((template,template))
#     xplot=np.linspace(0,2,profplot.shape[0])
#     ax1.plot(xplot, profplot, 'k')
#     ax1.plot(xplot, profplot2, 'r', linestyle='--')

#     ax1.set_xlabel('phase', fontsize=16)
#     ax1.set_xlim(0,2)

#     # Plot dynamic spectrum image
#     ax2.imshow(dspec_plot.T, aspect='auto', vmax=7, vmin=-2, origin='lower',
#                 extent=[0,T/60.,min(freqs), max(freqs)], cmap='Greys')
#     ax2.set_xlabel('time (min)', fontsize=16)
#     ax2.set_ylabel('freq (MHz)', fontsize=16)
#     ax2.yaxis.tick_right()
#     ax2.yaxis.set_label_position("right")

#     # Plot Secondary spectrum
#     ax3.imshow(SSb.T, aspect='auto', vmin=slow, vmax=shigh, origin='lower',
#                extent=[min(ft), max(ft), min(tau), max(tau)], interpolation='nearest',
#               cmap='Greys',
#               norm=LogNorm())
#     ax3.plot(ft,(((ft*u.mHz)**2)*eta_est).to_value(u.us),'r',alpha=.5)
#     ax3.plot(ft,(((ft*u.mHz)**2)*(eta_est+eta_high)).to_value(u.us),'r',alpha=.5)
#     ax3.plot(ft,(((ft*u.mHz)**2)*(eta_est-eta_low)).to_value(u.us),'r',alpha=.5)
#     if not eta_true==None:
#         ax3.plot(ft,(((ft*u.mHz)**2)*eta_true).to_value(u.us),'c',alpha=.5)
#     ax3.set_xlabel(r'$f_{D}$ (mHz)', fontsize=16)
#     ax3.set_ylabel(r'$\tau$ ($\mu$s)', fontsize=16) 

#     ax3.set_xlim(-xlim, xlim)
#     ax3.set_ylim(0, ylim*max(tau))

#     # Plot Hough Transform
#     ax4.loglog(etas[HT>0],HT[HT>0]/HT.max(),'r.')
#     ax4.axvline(eta_est.value,color='r')
#     ylim4=ax4.get_ylim()
#     if not eta_true==None:
#         ax4.axvline(eta_true.value,color='c')
#     ax4.set_ylabel('Hough Transform')
#     ax4.set_xlabel(r'$\eta$ ($ms/mHz^2$)' )
#     ax4.set_ylim(ylim4)
#     HG=half_gau(res.x,etas[HT>0])
#     ax4.loglog(etas[HT>0],HG,'k')

#     t0=Time(times.mean(),format='mjd')
#     ax1.set_title('                                                 {0}, {1}'.format(srce, t0.isot),
#                  fontsize=18)
#     return(eta_est,eta_low,eta_high)

def eta_from_data(dynspec,freqs,times,rbin=1,xlim=30,ylim=1,tau_lim=.001*u.ms,srce='',eta_true=None,prof=np.ones(10),template=np.ones(10)):
    nf=dynspec.shape[1]
    nt=dynspec.shape[0]
    df = (freqs[1]-freqs[0])*u.MHz
    dt = (((times[-1]-times[0]) / nt)*u.day).to(u.s)
    T=(dt*nt).value
    
    dspec=np.copy(dynspec)
    #if taper (Use Tukey window to taper edges of dynspec)
    t_window = scipy.signal.windows.tukey(dynspec.shape[0], alpha=0.2, sym=True)
    dspec *= t_window[:,np.newaxis]
    f_window = scipy.signal.windows.tukey(dynspec.shape[1], alpha=0.2, sym=True)
    dspec *= f_window[np.newaxis,:]

    #if pad (ADD PADDING, MASK REMOVAL)
    dspec = np.pad(dspec,((0,nt),(0,nf)),mode='constant',constant_values=0)
    SS = np.fft.fft2(dspec)/np.sqrt(nf*nt)
    SS = np.fft.fftshift(SS)
    SS = abs(SS)**2.0
    
    bintau = int(SS.shape[1] // rbin)

    SSb = SS.reshape(-1,SS.shape[1]//bintau, bintau).mean(-1)

    # Calculate the confugate frequencies (time delay, fringe rate), only used for plotting
    ft = np.fft.fftfreq(SS.shape[0], dt)
    ft = np.fft.fftshift(ft.to(u.mHz).value)

    tau = np.fft.fftfreq(SS.shape[1], df)
    tau = np.fft.fftshift(tau.to(u.microsecond).value)

    slow = np.median(SSb)*10**(-0.3)
    shigh = np.max(SSb)*10**(-1.5)

    # Hough Transform
    etas,HT,sig_low,sig_high=Hough_Rob(SS.T,tau,ft,rbin,tau_lim)
#     eta_est=etas[HT==HT.max()][0]
#     eta_low=eta_est-etas[HT>HT.max()*np.exp(-1./2)].min()
#     eta_high=etas[HT>HT.max()*np.exp(-1./2)].max()-eta_est
#     if eta_low==0:
#         eta_low=eta_est-etas[etas<eta_est].max()
#     if eta_high==0:
#         eta_high=etas[etas>eta_est].min()-eta_est
#     print(eta_high)
#     bounds=((0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf))
#     res=minimize(hg_fit,np.array([eta_est.value,1,eta_low.value,eta_high.value]),args=(etas[HT>0],HT[HT>0]/HT.max()),bounds=bounds)

#     eta_est,Amp,eta_low,eta_high=res.x
#     eta_est*=u.ms/u.mHz**2
#     eta_high*=u.ms/u.mHz**2
#     eta_low*=u.ms/u.mHz**2
    
    eta_est=etas[HT==HT.max()].mean()
#     eta_low=eta_est-etas[HT>=.99*HT.max()].min()
#     eta_high=etas[HT>=.99*HT.max()].max()-eta_est
    if sig_high[HT==HT.max()].max()==np.inf:
        eta_est=sig_low[sig_high==np.inf].max()
        eta_high=np.inf
        eta_low=0*u.ms/u.mHz**2
    else:
        eta_low=(sig_high[HT==HT.max()].max()-sig_low[HT==HT.max()].min())/2
        eta_high=(sig_high[HT==HT.max()].max()-sig_low[HT==HT.max()].min())/2
    print(eta_high)
    nbin = dynspec.shape[0]//2
    dspec_plot = dynspec[:nbin*2].reshape(nbin, 2, dynspec.shape[-1]).mean(1)
    plt.figure(figsize=(10,15))
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    plt.subplots_adjust(wspace=0.1)

    profplot=np.concatenate((prof,prof))
    profplot2=np.concatenate((template,template))
    xplot=np.linspace(0,2,profplot.shape[0])
    ax1.plot(xplot, profplot, 'k')
    ax1.plot(xplot, profplot2, 'r', linestyle='--')

    ax1.set_xlabel('phase', fontsize=16)
    ax1.set_xlim(0,2)

    # Plot dynamic spectrum image
    ax2.imshow(dspec_plot.T, aspect='auto', vmax=7, vmin=-2, origin='lower',
                extent=[0,T/60.,min(freqs), max(freqs)], cmap='Greys')
    ax2.set_xlabel('time (min)', fontsize=16)
    ax2.set_ylabel('freq (MHz)', fontsize=16)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    # Plot Secondary spectrum
    ax3.imshow(SSb.T, aspect='auto', vmin=slow, vmax=shigh, origin='lower',
               extent=[min(ft), max(ft), min(tau), max(tau)], interpolation='nearest',
              cmap='Greys',
              norm=LogNorm())
    ax3.plot(ft,(((ft*u.mHz)**2)*eta_est).to_value(u.us),'r',alpha=.5)
    ax3.plot(ft,(((ft*u.mHz)**2)*(eta_est+eta_high)).to_value(u.us),'r',alpha=.5)
    ax3.plot(ft,(((ft*u.mHz)**2)*(eta_est-eta_low)).to_value(u.us),'r',alpha=.5)
    if not eta_true==None:
        ax3.plot(ft,(((ft*u.mHz)**2)*eta_true).to_value(u.us),'c',alpha=.5)
    ax3.set_xlabel(r'$f_{D}$ (mHz)', fontsize=16)
    ax3.set_ylabel(r'$\tau$ ($\mu$s)', fontsize=16) 

    ax3.set_xlim(-xlim, xlim)
    ax3.set_ylim(0, ylim*max(tau))

    # Plot Hough Transform
    ax4.loglog(etas[HT>0],HT[HT>0]/HT.max(),'r.')
    ax4.axvline(eta_est.value,color='r')
    ax4.axvline((eta_est+eta_high).value,color='r')
    ax4.axvline((eta_est-eta_low).value,color='r')
    ylim4=ax4.get_ylim()
    if not eta_true==None:
        ax4.axvline(eta_true.value,color='c')
    ax4.set_ylabel('Hough Transform')
    ax4.set_xlabel(r'$\eta$ ($ms/mHz^2$)' )
    ax4.set_ylim(ylim4)

    t0=Time(times.mean(),format='mjd')
    ax1.set_title('                                                 {0}, {1}'.format(srce, t0.isot),
                 fontsize=18)
    return(eta_est,eta_low,eta_high)