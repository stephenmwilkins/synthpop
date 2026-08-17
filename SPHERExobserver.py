import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from synthesizer.filters import FilterCollection,Filter
from unyt import angstrom

#from Cartesian to sphericals point=[x,y,z] can be an array of namy points e.g. [[x1,y1,z1],[x2,y2,z2],...]
def cart2sph(points): 
    x=points[:,0]
    y=points[:,1]
    z=points[:,2]
    r=np.sqrt(x**2+y**2+z**2)+1e-12 #Just to prevent divide by 0 errors
    tta=np.arctan2(y,x)
    thi=np.arccos(z/r)
    return np.stack([r,tta,thi],axis=-1)

#from sphericals to cartesian point=[r,theta,phi] , can be an array of many points e.g. [[r1,theta1,phi1],[r2,theta2,phi2],...]
def sph2cart(points): 
    r=points[:,0]
    tha=points[:,1]
    phi=points[:,2]
    x=r*np.sin(phi)*np.cos(tha)
    y=r*np.sin(phi)*np.sin(tha)
    z=r*np.cos(phi)
    return np.stack([x,y,z],axis=-1)

#makes a new point "newOG" the new origin for points,  3D cartesian only
def grid_transf(points,newOG): 
    x=points[:,0]
    y=points[:,1]
    z=points[:,2]
    x_new=x-newOG[0]
    y_new=y-newOG[1]
    z_new=z-newOG[2]
    return np.stack([x_new,y_new,z_new],axis=-1)

#Plots frequency binned mollview projection for spherical coordinates points=[[theta1.phi1],[theta2,phi2],...]
def plot_pixel_mollview(points,NSIDE,title="Mollview",unit="Pixel Count"): 
    Theta=points[:,0]
    Phi=points[:,1]
    npix=hp.nside2npix(NSIDE)
    pix=hp.ang2pix(NSIDE,Theta,Phi)
    map=np.bincount(pix,minlength=npix)
    hp.mollview(map,title=title,unit=unit)

#Healpy Conversion, converts the cart2sph spherical coordinates to have ranges phi[0,2pi] and theta[0,pi] for use with healpy
#Need to use before you plot with plot_pixel_mollview
def healpy_conversion(points):
    Theta=points[:,1]
    Phi=points[:,2]
    Theta_for_healpy = Phi                        
    Phi_for_healpy = np.mod(Theta, 2 * np.pi)
    return np.stack([Theta_for_healpy,Phi_for_healpy],axis=-1)

# Makes a 3d grid with "size"
def test_grid(size): 
    line=np.arange(-size,size,1)
    X,Y,Z=np.meshgrid(line,line,line,indexing="xy")
    points=np.stack([X.ravel(),Y.ravel(),Z.ravel()],axis=-1)
    return points

#makes a random test grid with point values in range [0,1], with Npoints with the usual format.
def rand_test_grid(Npoints):
    rand=np.random.rand(Npoints,3)
    return rand

#Makes a random test grid using theta and phi only with Npoints. Outputs in format [[theta1,phi1],[theta2,phi2],...]
def rand_sph_grid(Npoints):
    r_theta=np.random.uniform(0,np.pi,Npoints)
    r_phi=np.random.uniform(0,2*np.pi,Npoints)
    return np.stack([r_theta,r_phi],axis=-1)

#plots a 3D grid
def plot_grid(points,figno=1,title="Plot"): 
    x=points[:,0]
    y=points[:,1]
    z=points[:,2]
    fig=plt.figure(figno)
    ax=plt.axes(projection="3d")
    ax.scatter(x,y,z,marker=".")
    fig.suptitle(title)

#returns the number of pixels on the SPHEREx sky
def SPHEREx_Npix():
    NPIX=((4*np.pi)*(648000)**2)/(6.2*np.pi)**2
    return NPIX


#returns an array of the SPHEREx observing wavebands in meters
def SPHEREx_photometry():
    wavebands=np.linspace(0.75,5.00,102)*1e-6
    return wavebands

#Creates a SPHEREx filter collection in synthesizer
def SPHEREx_instrument():
    wls=SPHEREx_photometry()*1e10*angstrom

    R41=np.ones(17)*41
    R35=np.ones(17)*35
    R110=np.ones(17)*110
    R130=np.ones(17)*130
    R=np.concatenate((R41,R41,R41,R35,R110,R130))


    filters=[]
    FWHMs=wls/R
    for n in range(101):
        filt=Filter(f"filter{n}",lam_eff=wls[n],lam_fwhm=FWHMs[n])
        filters.append(filt)

    SPHEREx=FilterCollection(filters=filters)
    
    return SPHEREx
