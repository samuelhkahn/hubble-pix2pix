import matplotlib.pyplot as plt
import numpy as np
def hist(data,nbins=100,fnx=10,fny=10):
    f = plt.figure(figsize=(fnx,fny))
    m = np.nanmean(data)
    s = np.nanstd(data)
    plt.hist(data.flatten(),bins=nbins,range=(-0.5*s,m+5*s),log=True,alpha=0.5)

def histcomp(data,datb,nbins=100,fnx=10,fny=10):
    f = plt.figure(figsize=(fnx,fny))
    m = np.nanmean(data)
    s = np.nanstd(data)
    plt.hist(data.flatten(),bins=nbins,range=(-0.5*s,m+5*s),log=True,alpha=0.5)
    plt.hist(datb.flatten(),bins=nbins,range=(-0.5*s,m+5*s),log=True,alpha=0.5)

def imshow(data,fnx=10,fny=10,flag_log=False,flag_cb=False,origin='lower'):
    f,ax = plt.subplots(1,1,figsize=(fnx,fny))
    pdata_tmp = data.copy()
    m = np.nanmean(pdata_tmp)
    if(flag_log==False):
        s = np.nanstd(pdata_tmp)
        vpmin = m - 0.5*s
        vpmax = m + 2.0*s
    else:
        vplmin = m/2.
        vpmin = np.log10(vplmin)
        vpmax = np.log10(m * 100.)
        pdata_tmp[pdata_tmp<vplmin] = vplmin
        pdata_tmp = np.log10(pdata_tmp)
    plt.imshow(pdata_tmp,vmin=vpmin, vmax=vpmax, origin=origin);
    if(flag_cb):
        plt.colorbar();
    return f,ax

def imshow_thumbnail(data,x,y,fnx=10,fny=10,nx=100,ny=100):

    # plot background-subtracted image
    fig, ax = plt.subplots(figsize=(fnx,fny))
    m, s = np.mean(data), np.std(data)
    dsplot = data.copy()
    ixmin = int(np.fmax(int(x-0.5*ny),0))
    ixmax = int(np.fmin(int(x+0.5*ny),data.shape[0]))
    jymin = int(np.fmax(int(y-0.5*nx),0))
    jymax = int(np.fmin(int(y+0.5*nx),data.shape[1]))
    im = ax.imshow(dsplot[jymin:jymax,ixmin:ixmax], interpolation='nearest', cmap='gray',
               vmin=m-s, vmax=m+s, origin='lower')
    plt.plot(0.5*nx,0.5*ny,'o',markersize=10,color='yellow')

def cutout(data,xmin,xmax,ymin,ymax):
    return data[xmin:xmax,ymin:ymax].copy()

def stats(data):
    print(f"Median = {np.nanmedian(data)}")
    print(f"Mean   = {np.nanmean(data)}")
    print(f"STDDEV = {np.nanstd(data)}")
    print(f"Max    = {np.nanmax(data)}")
    print(f"Min    = {np.nanmin(data)}")
