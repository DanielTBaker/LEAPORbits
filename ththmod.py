import numpy as np
import astropy.units as u
from scipy.linalg import eigh
from scipy.optimize import curve_fit

def chi_par(x,A,x0,C):
    return(A*(x-x0)**2+C)

def thth_err(N,eta,edges):
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    th1 = np.ones((th_cents.shape[0], th_cents.shape[0])) * th_cents
    th2 = th1.T
    err=np.ones(th1.shape)*np.sqrt(N)
    err*=np.abs(2*eta*(th2-th1)).value
    err=err**2
    return(err)

def thth_map(SS, tau, fd, eta, edges,fill_value=0):
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    th1 = np.ones((th_cents.shape[0], th_cents.shape[0])) * th_cents
    th2 = th1.T
    tau_inv = ((eta.value * (th1**2 - th2**2) /
                tau[1].value)).round().astype(int)
    fd_inv = (((th1 - th2) / fd.value[1])).round().astype(int)
    thth = np.ones(tau_inv.shape,dtype=complex)*fill_value

    pnts = (np.abs(tau_inv) < tau.shape[0]/2) * (np.abs(fd_inv) < fd.shape[0]/2)
    print(eta,tau_inv[pnts].max(),tau_inv[pnts].min(),fd_inv[pnts].max(),fd_inv[pnts].min())
    
    thth[pnts] = SS[tau_inv[pnts], fd_inv[pnts]]
    
#     uni,inv,coun=np.unique(fd_inv+1j*tau_inv,return_counts=True,return_inverse=True)
#     norm=coun[np.reshape(inv,(th1.shape[0],th1.shape[1]))]
#     thth/=norm
    thth*=np.abs(2*eta*(th2-th1)).value
    thth-=np.tril(thth)
    thth+=np.conjugate(np.triu(thth).T)
    thth-=np.diag(np.diag(thth))
    thth-=np.diag(np.diag(thth[::-1,:]))[::-1,:]
    thth=np.nan_to_num(thth)
    return (thth)

def rev_map(thth,tau,fd,eta,edges):
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    th1 = ((tau[:,np.newaxis]/eta + fd[np.newaxis,:]**2)/(2*fd[np.newaxis,:])).value
    th2 = th1 - fd[np.newaxis,:].value
    th1 -= edges[0] - np.diff(edges).mean()/2
    th1/=np.diff(edges).mean()
    th2 -= edges[0] - np.diff(edges).mean()/2
    th2 /= np.diff(edges).mean()
    th1 = np.floor(th1).astype(int)
    th2 = np.floor(th2).astype(int)
    pnts = (th1 > 0) * (th2>0) * (th1 < edges.shape[0]-1)  * (th2 < edges.shape[0]-1)
    recov=np.zeros((tau.shape[0],fd.shape[0]),dtype='complex')
    th_dif=th_cents[:,np.newaxis]-th_cents[np.newaxis,:]
    recov[pnts] = thth[th2[pnts],th1[pnts]]/np.abs(2*eta*th_dif[th2[pnts],th1[pnts]]).value
    recov = np.nan_to_num(recov)
    return(recov)
    
def rev_map_1d(V,tau,fd,eta,edges):
    th_cents = (edges[1:] + edges[:-1]) / 2
    th_cents -= th_cents[np.abs(th_cents) == np.abs(th_cents).min()]
    tau_map=((eta*(th_cents*fd.unit)**2+tau[1]/2)//tau[1]).astype(int)
    fd_map=((th_cents*fd.unit+fd[1]/2)//fd[1]).astype(int)
    SS=np.zeros((tau.shape[0],fd.shape[0]),dtype=complex)
    counts=np.zeros(SS.shape)
    for i in range(th_cents.shape[0]):
        SS[tau_map[i],fd_map[i]]+=V[i]
        counts[tau_map[i],fd_map[i]]+=1
    SS[counts>0]/=counts[counts>0]
    return(SS)
    
def eta_full(SS,fd,tau,mask,SS_red,fd_red,tau_red,mask_red,fd_lim,eta_low,eta_high,nth1,nth2,neta1,neta2,thth_fit=True):
    ##Determine Noise in SS
    N=(np.abs(SS[np.abs(tau)>5*tau.max()/6,:][:,np.abs(fd)>5*fd.max()/6])**2).mean()
    N2=SS_red[np.abs(tau_red)>5*tau_red.max()/6,:][:,np.abs(fd_red)>5*fd_red.max()/6].var()
    
    ##Setup Arrays for Amplitdue Search
    edges=np.linspace(-fd_lim,fd_lim,nth1)
    th_cents=(edges[1:]+edges[:-1])/2
    th_cents-=th_cents[np.abs(th_cents)==np.abs(th_cents).min()]
    etas_abs=np.linspace(eta_low.value,eta_high.value,neta1)*eta_low.unit
    chisq_abs=np.zeros(etas_abs.shape)
    dof_abs=np.zeros(etas_abs.shape)
    
    SS_red2=SS_red-SS_red[np.abs(tau_red)>5*tau_red.max()/6,:][:,np.abs(fd_red)>5*fd_red.max()/6].mean()

    ##Calculate chisq for each eta
    for i in range(etas_abs.shape[0]):
        eta=etas_abs[i]

        ##Find thth plot
        thth=thth_map(SS_red2, tau_red, fd_red, eta, edges,fill_value=0)
        ##Reduce to largest full square THTH matrix
        th_pnts=((th_cents**2)*eta.value<np.abs(tau.max().value)) * (np.abs(th_cents)<np.abs(fd.max()).value)
        thth_red=thth[th_pnts,:][:,th_pnts]
        edges_red=th_cents[th_pnts]
        edges_red=(edges_red[:-1]+edges_red[1:])/2
        edges_red=np.concatenate((np.array([edges_red[0]-np.diff(edges_red).mean()]),
                                  edges_red,
                                  np.array([edges_red[-1]+np.diff(edges_red).mean()])))
        ##Calculate Eigenvectors/values
        w,V=eigh(thth_red)
        ##Use larges eigenvector/value as model
        thth2_red=np.outer(V[:,np.abs(w)==np.abs(w).max()],np.conjugate(V[:,np.abs(w)==np.abs(w).max()]))
        thth2_red*=np.abs(w[np.abs(w)==np.abs(w).max()])
        thth2_red[thth_red==0]=0
        if thth_fit:
            ##Find error in THTH
            err=thth_err(N2,eta,edges)
            err_red=err[th_pnts,:][:,th_pnts]

            ##Calculate chi^2
            chisq_abs[i]=((np.abs(thth2_red-thth_red)**2)/err_red)[err_red>0].sum()
            chisq_abs[i]+=(np.abs(thth[err>0])**2/err[err>0]).sum() - (np.abs(thth_red[err_red>0])**2/err_red[err_red>0]).sum() 

            dof_abs[i]=err_red[err_red>0].shape[0]-1
        else:
            SS_rev=rev_map(thth2_red,tau_red,fd_red,eta,edges_red)
            chisq_abs[i]=np.sum(((np.abs(SS_rev-SS_red2)**2)/N2)[mask_red])
            dof_abs[i]=(SS_red2[tau_red>0,:].shape[0]*SS_red2[tau_red>0,:].shape[1])-1
        
    ##Find Region Around minimum
    print(etas_abs,chisq_abs)
    C=chisq_abs.min()
    x0=etas_abs[chisq_abs==C][0].value
    A=C/(etas_abs[1]-etas_abs[0]).value**2
    etas_fit=etas_abs[np.argsort(chisq_abs)][:20]
    chisq_fit=chisq_abs[np.argsort(chisq_abs)][:20]

    chisq_fit=chisq_fit[np.argsort(etas_fit)]
    etas_fit=etas_fit[np.argsort(etas_fit)]

    popt,pcov=curve_fit(chi_par,etas_fit.value,chisq_fit,p0=np.array([A,x0,C]))


    eta=popt[1]
    eta_sig=np.sqrt((chisq_fit-chi_par(etas_fit.value,*popt)).std()/popt[0])

    ##Setup Arrays for Complex Search
    edges=np.linspace(-fd_lim,fd_lim,nth2)
    th_cents=(edges[1:]+edges[:-1])/2
    th_cents-=th_cents[np.abs(th_cents)==np.abs(th_cents).min()]
    eta_low=eta-2*eta_sig
    etas=np.linspace(eta_low,eta+2*eta_sig,neta2)
    if eta_low<0:
        eta_low=etas[etas>0].min()
    etas=np.linspace(eta_low,eta+2*eta_sig,neta2)*eta_low.unit
    chisq=np.zeros(etas.shape)
    dof=np.zeros(etas.shape)
    
    ##Calculate chisq for each eta
    for i in range(etas.shape[0]):
        eta=etas[i]

        ##Find thth plot
        thth=thth_map(SS, tau, fd, eta, edges,fill_value=0)
        ##Reduce to largest full square THTH matrix
        th_pnts=((th_cents**2)*eta.value<np.abs(tau.max().value)) * (np.abs(th_cents)<np.abs(fd.max()).value)
        thth_red=thth[th_pnts,:][:,th_pnts]
        edges_red=th_cents[th_pnts]
        edges_red=(edges_red[:-1]+edges_red[1:])/2
        edges_red=np.concatenate((np.array([edges_red[0]-np.diff(edges_red).mean()]),
                                  edges_red,
                                  np.array([edges_red[-1]+np.diff(edges_red).mean()])))
        ##Calculate Eigenvectors/values
        w,V=eigh(thth_red)
        ##Use larges eigenvector/value as model
        thth2_red=np.outer(V[:,np.abs(w)==np.abs(w).max()],np.conjugate(V[:,np.abs(w)==np.abs(w).max()]))
        thth2_red*=np.abs(w[np.abs(w)==np.abs(w).max()])
        thth2_red[thth_red==0]=0
        if thth_fit:
            ##Find error in THTH
            err=thth_err(N,eta,edges)
            err_red=err[th_pnts,:][:,th_pnts]

            ##Calculate chi^2
            chisq[i]=((np.abs(thth2_red-thth_red)**2)/err_red)[err_red>0].sum()
            chisq[i]+=(np.abs(thth[err>0])**2/err[err>0]).sum() -( np.abs(thth_red[err_red>0])**2/err_red[err_red>0]).sum() 
            dof[i]=err_red[err_red>0].shape[0]-1
        else:
            SS_rev=rev_map(thth2_red,tau,fd,eta,edges_red)
            chisq[i]=np.sum(((np.abs(SS_rev-SS)**2)/N)[mask])
            dof[i]=(SS[tau>0,:].shape[0]*SS[tau>0,:].shape[1])-1
    try:
        chisq_fit=chisq[np.abs(etas-etas[chisq==chisq.min()])<np.diff(etas).mean()*10]
        etas_fit=etas[np.abs(etas-etas[chisq==chisq.min()])<np.diff(etas).mean()*10]

        chisq_fit=chisq_fit[np.argsort(etas_fit)]
        etas_fit=etas_fit[np.argsort(etas_fit)]
        x0=etas_fit[chisq_fit==chisq_fit.min()][0].value
        C=chisq_fit.min()
        A=np.nan_to_num((chisq_fit-C)/(etas_fit.value-x0)**2).max()
        popt,pcov=curve_fit(chi_par,etas_fit.value,chisq_fit,p0=(A,x0,C))

        eta_fit=popt[1]*eta_low.unit
        eta_sig=(1/np.sqrt(popt[0]))*eta_low.unit
    except:
        eta_sig=None
        eta_fit=None
    return(etas_abs,chisq_abs,etas,chisq,eta_fit,eta_sig)