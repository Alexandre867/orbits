# Copyright (c) 2022 Alexandre Daigneault
# 
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
orbit v1.03

Provides 'orbit' class objects and space() methods.
Orbits and ground tracks assuming restricted 2-body problem conditions.
Uses numpy, scipy, pandas, and matplotlib libraries.
Also creates a 'mercator' scale for matplotlib (from matplotlib examples).

Example use:

from orbits import *
R=[8400,0,0]
V=[0,5.4,7.2]
o1 = orbit(R=R,V=V)
describe_orbit(o1)
space().plot_ground_track(o1,t=2.5*o1.P,dt=2.5*o1.P/500,time=True,alt=True,figsize=(14,10),facecolor='white',dpi=100,cmap='jet') #turbo colormap preferable if available

##### Version 1.01 #####

%matplotlib auto
# For Jupyter Notebook:
# %matplotlib widget
o1 = orbit(a=1e4,e=.3,i=40,Omega=0,omega=30,f=120)
describe_orbit(o1)
o2 = orbit(a=1e4,e=.2,i=20,Omega=90,omega=180,f=230)
describe_orbit(o2)
space().plot3d_orbit([o1,o2])

##### Version 1.02 #####

%matplotlib auto
# For Jupyter Notebook:
# %matplotlib widget
o1 = orbit(a=1e4,e=.3,i=40,Omega=0,omega=30,f=60)
o2 = orbit(a=1.1e4,e=.2,i=20,Omega=90,omega=180,f=230)
# Transfer from orbit o1 to orbit o2 using perpendicular impulse for inclination and phasing in two revolutions.
data=space().orbit_transfer(o1,o2,direct_incl=False,phasing=True,n=2,minR=6371+100,asap=False,output=None,full_output=False)
display(data.iloc[:,:-1])
space().plot_orbit_transfer(data,plot=True,dim=3)

---------- Update log 1.01 ----------
orbit class:
    - An orbit instance can now have its properties updated through assignment,
    e.g. orbit_instance.a=1e4, it will update the other dependant properties.
    The update of the other properties is announced if update_message==True.
    
    - mu is now read-only.
    
    - hash, ==, != bult-in functions now implemented.
    
    - orbit instance can be used as an iterator and will return itself once.
    
    - copy method has been updated.
    
    - E, M, ap, pe set as properties instead of methods
    (i.e. use orbit_instance.M instead of orbit_instance.M())

space class:
    - calc_rv method can now take in a list of values for f (true anomaly).
    R and V are returned as horizontal vectors (stacked vertically if f is a list).
    
    - plot3d_orbit: NEW METHOD FOR 3D PLOTTING ORBITS! Using wireframe for the
    planet: big issue with ordering of surface vs line otherwise and fixing it
    would require using a different library

---------- Update log 1.02 ----------
    - s2hms and hms2s: new functions to convert between seconds and h,m,s
    
    - proj_on: new function to calculate the projection of a vector on a subspace
    spanned by given vectors.

orbit class:
    - == overload modified.
    
    - len built-in function implemented.
    
    - dV: new method for tangential impulses.

space class:
    - plot3d_orbit now displays velocity vector of the object. Length of vector
    can be adjusted using vscale. plot3d_orbit now defaults to 'ortho' (orthogonal)
    projection instead of 'persp' (perspective) projection. Orbs can be a list
    of lists, with the other arguments in the list specifying the final true anomaly
    and possibly the initial anomaly.
    
    - plot2d_orbit: new method that plots the given orbits in 2D by ignoring one
    of the dimensions.
    
    - plot3d_flat_orbit: new method that plots the given orbits using 2D
    orthographic projection.
    
    - R2f: new method that returns the true anomaly on the orbit where the position
    is in the same direction as the given vector.
    
    - orb_plane_inters: new method that returns the true anomaly on the first
    orbit which is at the intersection of the planes of the two given orbits.
    
    - trans_orb_plane: method that returns a new orbit identical in size and shape
    as the first orbit but lying on the plane of the second orbit.
    
    - orb_inters: new method that returns one intersection of the two orbits as
    a direction phi=f+omega.
    
    - orb_paral: new method that returns one direction (phi=f+omega) where the
    two orbits are parallel.
    
    - scale_burn: new method that returns how much to scale the velocity at the
    current position to make the orbit tangential to the other orbit. Best to use
    somewhere where the orbits are parallel.
    
    - phasing_orbits: new method that returns the delta V for coorbital phasing.
    
    - orbit_transfer: new method that calculates the transfer between two orbits.
    
    - plot_orbit_transfer: new method that plots the data returned by orbit_transfer
    or make it ready to be plotted.
    
----- Update log 1.03 -----
    - Correction of bugs
    
space class:
    - orbit_transfer: Output with full_output=True modified: now returns 4 args:
    output,useful,deltaV_tot,dist, where dist is the distance from the target
    it reaches. dist=None if phasing=False.

"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import fsolve#, root
import warnings

class InitError(Exception):
    """
    Custom error for intialisation error
    """
    pass

def ang_bet(A,B):
    """
    Angle in RADIANS between vectors A and B. (Vectors as numpy arrays)
    """
    return np.arccos(np.clip(np.dot(A,B)/(norm(A)*norm(B)), -1.0, 1.0))

def proj_on(V_i,*args):
    """
    Return the projection of vector V_i on the subspace spanned by the vectors
    in args.
    """
    array_args = np.vstack([args]).T
    if len(args)>np.linalg.matrix_rank(array_args): raise ArithmeticError('Vectors in args must be linearly independant.')
#     if len(V_i)==np.linalg.matrix_rank(array_args): return V_i
    ortho_args = np.linalg.qr(array_args)[0].T
    return ortho_args @ V_i @ ortho_args

def s2hms(st):
    """
    Takes in time in seconds, returns hour,min,sec.
    """
    s=st%60
    st=st//60
    m=int(st%60)
    h=int(st//60)
    return h,m,s

def hms2s(h,m,s):
    """
    Takes in hour,min,sec, returns time in seconds.
    """
    return h*3600+60*m+s

import pandas as pd
def describe_orbit(orbit):
    """
    To print out nicely the orbital parameters and state vector of an 'orbit' object.
    The only method that uses pandas library.
    """
    print(pd.Series(orbit.coe(),index=['a','e','i','Ω','ω','ν']),'\n')
    print(pd.DataFrame(list(orbit.rv()),index=['R','V'],columns=['x','y','z']))

# class planet:
#     """
#     This method is not used.
#     """
#     def __init__(self,mu=None,M=None,R=0,X=0,Y=0):
#         self._G=6.67e-11
#         if mu is not None:
#             self.mu=mu
#             if M is not None: self.M=M
#             else: self.M=mu/self._G
#         elif M is not None:
#             self.M=M
#             if mu is not None: self.mu=mu
#             else: self.mu=M*self._G
#         else: raise InitError("Missing information: mu and M undefined")
#         self.R=R
#         self.X=X
#         self.Y=Y

class orbit:
    """
    Defines the 'orbit' object.
    Initialised with either state vectors or classical orbital elements.
    By default, a circular equatorial orbit with semi-major axis of 7000km is initialised.
    
    Angular variables must be in degrees [°].
    Units must be consistent with mu, which by default is in [km] and [s].
    a: semi-major axis (default: 7000 [km])
    e: excentricity (must be >0)
    i: inclination [°]
    Omega: RAAN, or longitude of right ascending node [°]
    omega: argument of periapsis [°]
    f: true anomaly [°]
    R: position vector (as a list or numpy array)
    V: velocity vector (as a list or numpy array)
    P: Orbital period, can be used to describe the orbit size; has priority over 'a'
    mu: gravitational parameter (default: 398600 [km^3/s^2] (mu for Earth))
    """
    
    def __setattr__(self,name,value):
        """
        Changing parameters through assignment such as orbit.a = 8000 will update the dependant elements.
        Note that changing a classical element and either R or V may produce something else than desired.
        """
        super(orbit, self).__setattr__(name, value)
        if self._update_elements and name != '_update_elements':
            self._update_elements = False
            self._regulate()
            if name == 'P':
                self.a=((self.P/(2*np.pi))**2*self.mu)**(1/3)
                if self.update_message: print('Updated a')
            if name in ['a','e','i','Omega','omega','f','P']:
                self.R,self.V=space().calc_rv(*self.coe(),self.mu)
                if self.update_message: print('Updated R and V')
            if name in ['R','V']:
                self.a,self.e,self.i,self.Omega,self.omega,self.f=space().calc_coe(*self.rv(),self.mu)
                self.P=2*np.pi*(self.a**3/self.mu)**(1/2)
                if self.update_message: print('Updated COE')
            if name == 'a':
                self.P=2*np.pi*(self.a**3/self.mu)**(1/2)
                if self.update_message: print('Updated P')
            self._update_elements = True
    
    @property
    def mu(self):
        """
        Make mu read-only
        """
        return self._mu
    
    def __init__(self,a=7000,e=0,i=0,Omega=0,omega=0,f=0,R=None,V=None,P=None,mu=398600):
        self._update_elements = False
        self.update_message = True
        self._mu=mu
        if R is not None and V is not None:
            self.R=R
            self.V=V
            self.a,self.e,self.i,self.Omega,self.omega,self.f=space().calc_coe(R,V,mu)
            self.P=2*np.pi*(self.a**3/mu)**(1/2)
        elif R is not None or V is not None:
            raise InitError("R or V undefined.")
        else:
            if P is not None:
                self.P=P                                #[s]
                self.a=((self.P/(2*np.pi))**2*self.mu)**(1/3)
            else:
                self.a=a
                self.P=2*np.pi*(self.a**3/mu)**(1/2)    #[s]
            self.e=e
            self.i=i
            self.Omega=Omega
            self.omega=omega
            self.f=f
            self.R,self.V=space().calc_rv(a,e,i,Omega,omega,f,mu)
        self._update_elements = True
        self._regulate()
    
    def __hash__(self):
        """
        For computing the instance's hash.
        """
        return hash(self.coe())
        
    def __eq__(self,other):
        """
        For == operator.
        """
        if isinstance(other, type(self)): return all(np.isclose(self.coe(),other.coe()))
        return False
    
    def __ne__(self,other):
        """
        For != operator.
        """
        return not(self==other)
    
    def __len__(self):
        """
        For len().
        """
        return 1
    
    def __iter__(self):
        """
        Allows orbit instance to be used as an iterator (e.g. for for-loops).
        """
        self._itered = False #For using orbit as iterator
        return self
    
    def __next__(self):
        """
        When orbit instance is used as an iterator, it returns itself once.
        """
        if self._itered: raise StopIteration
        self._itered = True
        return self
    
    def _regulate(self):
        """
        Internal method used to keep the values of the object's parameters within
        correct bounds bounds.
        """
        restore_update_elements = self._update_elements
        self._update_elements = False
#         assert self.a>0, "Negative semi-major axis not supported yet"
        assert self.e>=0, "Invalid excentricity: %s"%(self.e)
#         assert self.e<1, "Orbit is not closed; excentricity > 1"
#         self.i=(self.i+90)%180-90
        if self.i<0: self.Omega=self.Omega+180
        self.i=np.rad2deg(np.arccos(np.cos(np.deg2rad(self.i))))    #So that 180° is valid
        self.Omega%=360
        self.omega%=360
        self.f%=360
        self._update_elements = restore_update_elements
    
    def coe(self):
        """
        Returns classical orbital parameters
        """
        return self.a,self.e,self.i,self.Omega,self.omega,self.f
    
    def rv(self):
        """
        Returns state vectors
        """
        return self.R,self.V
    
    def copy(self):
        """
        Returns copy of itself
        """
#         return orbit(*self.coe(),mu=self.mu)
        copied = orbit()
        copied.__dict__ = self.__dict__.copy()
        return copied
    
    def update(self,a=None,e=None,i=None,Omega=None,omega=None,f=None,R=None,V=None,P=None):
        """
        Serves to change several parameters of the orbit.
        Does not allow to change the value for mu, as it would be ambiguous whether to keep the
        classical orbital elements or the state vectors. It is better to use a new 'orbit' instance.
        'P' has again priority over 'a'
        Classical orbital elements have priority over state vectors
        Returns the updated 'orbit'
        """
        restore_update_elements = self._update_elements
        self._update_elements = False
        if P is not None: a=((P/(2*np.pi))**2*self.mu)**(1/3)
        keys1={'a':a,'e':e,'i':i,'Omega':Omega,'omega':omega,'f':f}
        keys2={'R':R,'V':V}
        if any([x is not None for x in keys1.values()]):
            for x in keys1.keys():
                if keys1[x] is not None:
                    setattr(self,x,keys1[x])
#                     self.__dict__[x]=keys1[x]
            self._regulate()
            self.R,self.V=space().calc_rv(*self.coe(),self.mu)
            self.P=2*np.pi*(self.a**3/self.mu)**(1/2)
        elif any([x is not None for x in keys2.values()]):
            for x in keys2.keys():
                if keys2[x] is not None:
                    setattr(self,x,keys2[x])
#                     self.__dict__[x]=keys2[x]
            self.a,self.e,self.i,self.Omega,self.omega,self.f=space().calc_coe(*self.rv(),self.mu)
            self.P=2*np.pi*(self.a**3/self.mu)**(1/2)
        self._update_elements = restore_update_elements
        return self
    
    def dV(self,dV):
        """
        Adds dV tangentially to the current velocity and return self.
        """
        return self.update(V=self.V/norm(self.V)*dV+self.V)
    
    @property
    def E(self):
        """
        Returns the eccentric anomaly
        """
        e=self.e
        f=np.deg2rad(self.f)
        E=np.arccos((e+np.cos(f))/(1+e*np.cos(f)))
        if f>np.pi: E=2*np.pi-E
        return np.rad2deg(E)
    
    @property
    def M(self):
        """
        Returns the mean anomaly
        """
        e=self.e
        E=np.deg2rad(self.E)
        return np.rad2deg(E-e*np.sin(E))
    
    @property
    def ap(self):
        """
        Returns the apoapsis
        """
        return self.a*(1+self.e)
    
    @property
    def pe(self):
        """
        Returns the periapsis
        """
        return self.a*(1-self.e)


class space:
    """
    Class for different methods making use of 'orbit' instances.
    """
    def __init__(self):
        pass
    
    def calc_coe(self,R,V,mu):
        """
        Returns the classical orbital elements for state vectors 'R' and 'V' and
        gravitational parameter 'mu'.
        """
        R=np.array(R)
        V=np.array(V)
        r=norm(R)
        v=norm(V)
        epsilon = v**2/2-mu/r
        a=-mu/(2*epsilon)
        E=1/mu*((v**2-mu/r)*R-sum(R*V)*V)
        e=norm(E)
        H=np.cross(R,V)         #vector of h
#         h=norm(H)
        i=np.rad2deg(ang_bet([0,0,1],H))
        N=np.cross([0,0,1],H)   #vector of n
        if norm(N)==0: N=np.array([1,0,0])
#         n=norm(N)
        Omega=np.rad2deg(ang_bet([1,0,0],N))
        if N[1]<0: Omega=(360-Omega)%360
        if e==0:
            omega=0
            E=N
        else:
            omega=np.rad2deg(ang_bet(N,E))
#             if E[2]<0: omega=360-omega            ####### NOT WORKING FOR i=0
#             if sum(E*np.cross([0,0,1],N))<0: omega=360-omega  ##### NOT WORKING FOR i=180
            if sum(E*np.cross(H,N))<0: omega=360-omega 
        f=np.rad2deg(ang_bet(E,R))
        if sum(R*V)<0: f=360-f
#         print(N)
#         print(E)
#         print(R)
        return a,e,i,Omega,omega,f
    
    def calc_rv(self,a,e,i,Omega,omega,f,mu):
        """
        Returns the state vectors 'R' and 'V' from the classical orbital elements
        and the gravitational parameter 'mu'.
        f can be a single value or a list.
        R and V returned as horizontal numpy array
        For a list of f, R and V returned as 2D array with R and V vectors being
        horizontal.
        """
        i=np.deg2rad(i)
        Omega=np.deg2rad(Omega)
        omega=np.deg2rad(omega)
        f=np.deg2rad(f)
        T_i=np.array([[1,0,0],
              [0,np.cos(i),-np.sin(i)],
              [0,np.sin(i),np.cos(i)]])
        T_Omega=np.array([[np.cos(Omega),-np.sin(Omega),0],
                          [np.sin(Omega),np.cos(Omega),0],
                          [0,0,1]])
        T_mat=np.matmul(T_Omega,T_i)
        r=a*(1-e**2)/(1+e*np.cos(f))
        v=np.sqrt(mu*(2/r-1/a))
        l=omega+f
        R=np.transpose(r*T_mat.dot(np.array([np.cos(l),np.sin(l),np.zeros_like(f)])))
        rp=a*(1-e**2)/(1+e*np.cos(f))**2*e*np.sin(f)    #dr/df
        V=np.array([np.cos(l)*rp-np.sin(l)*r,
                    np.sin(l)*rp+np.cos(l)*r,
                    np.zeros_like(f)])
        V=V/norm(V)*v
        V=np.transpose(T_mat.dot(V))
        return(R,V)
        
    def predict(self,orbit,t,rv=True):
        """
        Returns position (if 'rv' is True) of an object after a given time interval.
        For 'rv' is False, returns the true anomaly.
        """
        t%=orbit.P
        e=orbit.e
#         f=np.deg2rad(orbit.f)
#         E=np.arccos((e+np.cos(f))/(1+e*np.cos(f)))
#         if f>np.pi: E=2*np.pi-E
#         M=E-e*np.sin(E)
        M=np.deg2rad(orbit.M)
        n=2*np.pi/orbit.P
        M=(M+n*t)%(2*np.pi)
        E=M
        iterations = 6
        for i in range(iterations): E-=(E-e*np.sin(E)-M)/(1-e*np.cos(E))    #Newton's method
        f=np.arccos((np.cos(E)-e)/(1-e*np.cos(E)))
        if E>np.pi: f=2*np.pi-f
        f=np.rad2deg(f)
        if rv: return orbit.copy().update(f=f).rv()
        return f
    

    def ground_track_pos(self,orbit,t,t_0=0,dt=None,time=True):
        """
        Returns the ground track for a given orbit and a given time interval as
        lists of longitude, latitude and distance R.
        Assumes that (0°N, 0°W) is aligned with [1,0,0] of the orbit's reference
        frame at t=0.
        
        orbit: orbit to track
        t: end of time range (units depending on mu of orbit, default: [s])
        t_0: beginning of time range (default: 0)
        dt: step interval for the reported positions; doesn't affect accuracy (default: (t-t_0)/1000)
        time: if True (default), t, t_0 and dt are given in time. Otherwise, they
        are taken as defining the range to report in degrees of true anomaly,
        with t_0=0 being the current position of the orbit object.
        """
        if dt is None: dt=(t-t_0)/1000
        if not(time):
            T=np.arange(orbit.f+t_0,orbit.f+t+t_0,dt)
            o2=orbit.copy()
            pos_cart=np.vstack(list(map(lambda x: o2.update(f=x).R,T)))
            T = (np.array(list(map(lambda x: o2.update(f=x).M,T)))-orbit.M)%360/(360/orbit.P)+(T-orbit.f)//360*orbit.P
        else:
            T=np.arange(t_0,t,dt)
            pos_cart=np.vstack(list(map(lambda x: self.predict(orbit,x,rv=True)[0],T)))
        
        lon = (np.rad2deg(np.arctan2(pos_cart[:,1],pos_cart[:,0]))-T*360/86164.0905+180)%360-180
#         lon-=lon[0]
        lat = 90-np.rad2deg(np.arctan2((pos_cart[:,0]**2+pos_cart[:,1]**2)**(1/2),pos_cart[:,2]))
        r = (pos_cart[:,0]**2+pos_cart[:,1]**2+pos_cart[:,2]**2)**(1/2)
        return lon,lat,r
    
    def collision_detect(self,orbit,R,time=True):
        """
        Dectects if an object on orbit collides with the planet of radius R.
        Returns None if no collision, 0 if object already under the surface, and time before impact if time=True (default), or the true anomaly at impact if time=False.
        """
        if orbit.pe>R: return None
        if norm(orbit.R)<R: return 0
        f=360-np.rad2deg(np.arccos((orbit.a*(1-orbit.e**2)/R-1)/orbit.e))
        if not time: return f
        return (orbit.copy().update(f=f).M-orbit.M)/(360/orbit.P)
    
#########################################################
    def plot_ground_track(self,orbit,t,t_0=0,dt=None,time=True,background=True,R=6378,alt=False,collide_end=False,savefig=None,title=None,xgrid_step=15,ygrid_step=15,vbox=[[-180,-82],[180,82]],cmap='jet',**kwargs):
        """
        Plots the ground track of an orbit. Assumes that (0°N, 0°W) is aligned with [1,0,0] of the orbit's reference frame at t=0.
        
        orbit, t, t_0, dt, time: same as used by ground_track_pos()
        background: Whether to put the world map in background or not. Turned off when not using the default view box.
        Reads it from ''Earth mercator projection.jpg'. Otherwise, using the one at: https://upload.wikimedia.org/wikipedia/commons/f/f4/Mercator_projection_SW.jpg
        R: Planet radius, used for altitude and collision detection (units depending on orbit's mu, default: 6378 [km] (Earth radius))
        alt: Whether to plot color as a function of altitude or radial distance (default: False)
        collide_end: Whether to stop plotting when the object collides with the planet (default: False)
        savefig: If not None, saves the picture at the specified path (note: can be saved as a pdf)
        title: title for the graph (default: None)
        Ex: 
        'a=%s km, e=%s, i=%s°, Ω=%s°, ω=%s°, ν=%s°'%(orbit.coe())
        'R=%s km, V=%s km/s'%(orbit.rv())
        xgrid_step, ygrid_step: intervals at which to display graduation and grids (note: doesn't show decimals)
        vbox: view box, defined by a list of size 2: coordinates of lower left corner, and coordinates of upper right corner.
        cmap: colormap used to plot the altitude or radial distance of the object (recommended: 'turbo', see https://gist.github.com/FedeMiorelli/640bbc66b2038a14802729e609abfe89 to add it as a colormap)
        **kwargs: other values to pass to pyplot.figure()
        Recommendation: figsize=(14,10),facecolor='white',dpi=200
        """
        if collide_end:
            collide = space().collision_detect(orbit,R,time=time)
            if collide is not None:
                if collide == 0: raise Exception('Collision detection: Object is already in the ground')
                if time: t = min(t,collide)
                else: t= min(t,(collide-orbit.f)%360)
        if not alt: R=0
        lon,lat,r=space().ground_track_pos(orbit=orbit,t=t,t_0=t_0,dt=dt,time=time)
        
        if vbox!=[[-180,-82],[180,82]]: background=False    #Background with changes of dimensions is not currently supported

        lon_range = [vbox[0][0],vbox[1][0]]
        lat_range = [vbox[0][1],vbox[1][1]]
        
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)
        if background:
            try:
                img = mpimg.imread('Earth mercator projection.jpg',format='jpg')
#                 imgplot = 
                ax.imshow(img, extent=(-181,181,np.deg2rad(-82.5),np.deg2rad(82.7)),aspect='auto')
            except:
                try:
                    img = mpimg.imread('https://upload.wikimedia.org/wikipedia/commons/f/f4/Mercator_projection_SW.jpg',format='jpg')
#                     imgplot = 
                    ax.imshow(img, extent=(-181,181,np.deg2rad(-82.5),np.deg2rad(82.7)),aspect='auto')
                except: pass
        ax.set_ylim((np.deg2rad(lat_range)))
        # ax.yaxis.clear()
        ax.yaxis._visible=False
        # ax.tick_params(left=False,labelleft=False)
        ax.grid(axis='x')
        ax=ax.twinx()
        ax.yaxis.tick_left()
        ax.set_yscale('mercator')
        # ax.set_title('a=%s km, e=%s, i=%s°, Ω=%s°, ω=%s°, ν=%s°'%(orbit.coe()))
        # ax.set_title('R=%s km, V=%s km/s'%(orbit.rv()))
        ax.set_title(title)
        scat = ax.scatter(lon[r>R],np.deg2rad(lat[r>R]),s=5,c=r[r>R]-R,cmap=cmap)
        cbar = fig.colorbar(scat)
        if alt: cbar.ax.set_ylabel('Altitude [km]')
        else: cbar.ax.set_ylabel('R [km]')
        ax.scatter(lon[r<R],np.deg2rad(lat[r<R]),s=5,c='black')
        ax.set_xlim(lon_range)
        ax.set_ylim((np.deg2rad(lat_range)))
        ax.grid(axis='y')
        fmt = FuncFormatter(
            lambda x, pos=None: f"{x:.1f}\N{DEGREE SIGN}")
        ax.xaxis.set(major_locator=FixedLocator(np.arange(-180, 181, xgrid_step)),
                         major_formatter=fmt, minor_formatter=fmt)
        
        fmt = FuncFormatter(
            lambda x, pos=None: f"{np.degrees(x):.1f}\N{DEGREE SIGN}")
        ax.yaxis.set(major_locator=FixedLocator(np.radians(np.arange(-90, 90, ygrid_step))),
                 major_formatter=fmt, minor_formatter=fmt)
        
        
        fig.tight_layout()
        if savefig is not None: fig.savefig(savefig)
        
        plt.show()
        
    def plot3d_orbit(self,orbs,R=6371,res=100,faces=20,planet_color=[0,.25,0],vscale=1e3,proj='ortho'):
        """
        Plots one or several orbits in 3D.

        orbs: One or several orbits to plot in a list
        R: Planet radius [km or same units as orbits/mu] (default: Earth radius)
        res: Resolution of the orbit calculation
        faces: Number of faces for planet
        planet_color: Color of the planet
        vscale: How much to multiply the velocity vectors to draw the arrows
        proj: projection to use {'persp','ortho'}
        """
        assert R>=0, 'Planet radius must be positive'
        
        f=np.linspace(0,360,res)

        phi = np.linspace(0,2*np.pi,faces+1)
        rho = np.linspace(0,np.pi,faces+1)
        phi,rho=np.meshgrid(phi,rho)
        X=np.sin(rho)*np.cos(phi)*R
        Y=np.sin(rho)*np.sin(phi)*R
        Z=np.cos(rho)*R
#         [X,Y,Z]=sphere(args.faces);
#         hold off
#         surf(X*R,Y*R,Z*R,np.repmat(np.reshape(planet_color,1,1,[]),np.shape(X,1),np.shape(X,2)))
        fig = plt.figure(figsize=(10,10))#,dpi=80)
        ax = Axes3D(fig,auto_add_to_figure=False)
        fig.add_axes(ax)
#         ax.plot_surface(X,Y,Z,color=planet_color,edgecolor='black',rstride=1,cstride=1,shade=False)
        ax.plot_wireframe(X,Y,Z,colors=planet_color,rstride=1,cstride=1)
#         hold on
#         axis equal
        
        minv,maxv=[-R,-R,-R],[R,R,R]
        for orb in orbs:
            if len(orb)==3:
                orb,minf,maxf = orb
            elif len(orb)==2:
                orb,maxf = orb
                minf = orb.f
            elif len(orb)==1:
                if isinstance(orb,list): orb=orb[0]
                minf=0
                maxf=360
            else: raise TypeError("Invalid input:",orb)
            if np.isclose(maxf%360,minf): minf,maxf=0,360
            else: maxf=(maxf-minf)%360+minf
            f=np.linspace(minf,maxf,res)
            R,_=space().calc_rv(*orb.coe()[:-1],f,orb.mu);
            minv=np.amin(np.vstack([R,minv]),axis=0)
            maxv=np.amax(np.vstack([R,maxv]),axis=0)
            ax.plot(R[:,0],R[:,1],R[:,2],linewidth=2.5)
            if (0-minf)%360<=(maxf-minf)%360 or (maxf-minf)%360==0:
                R,_=space().calc_rv(*orb.coe()[:-1],0,orb.mu);
                ax.plot(R[0],R[1],R[2],'s',ms=10,mew=2,mec='lime',mfc='white')
            if (180-minf)%360<=(maxf-minf)%360 or (maxf-minf)%360==0:
                R,_=space().calc_rv(*orb.coe()[:-1],180,orb.mu);
                ax.plot(R[0],R[1],R[2],marker='s',ms=10,mew=2,mec='r',mfc='white')
            R,V=orb.rv()
            V=V*vscale
            ax.plot(R[0],R[1],R[2],marker='o',ms=10,mew=2,mec='b',mfc=[0,0,0,0])
#             ax.quiver3D(R[0],R[1],R[2],V[0],V[1],V[2],length=vscale)
            V=R+V
#             minv=np.amin(np.vstack([V,minv]),axis=0)
#             maxv=np.amax(np.vstack([V,maxv]),axis=0)
            arrow = Arrow3D([R[0], V[0]], [R[1], V[1]], 
                        [R[2], V[2]], mutation_scale=20, #Arrow head size
                        lw=3, arrowstyle="-|>", color="r")
            ax.add_artist(arrow)
#         axis equal
#         ax.axis('equal')
#         ax.set_aspect(1)
        ax.set_xlim(minv[0],maxv[0])
        ax.set_ylim(minv[1],maxv[1])
        ax.set_zlim(minv[2],maxv[2])
        ax.set_box_aspect(np.ptp(np.vstack([minv,maxv]),axis=0))
        ax.set_proj_type(proj)

        plt.show()
        
    def plot2d_orbit(self,orbs,axis='z',R=6371,res=100,planet_color=[0,.25,0],vscale=1e3,fig=None,ax=None,**kwargs):
        """
        Plots the orbits in orbs in 2D by ignoring the axis specified by axis.
        fig,ax: figure and axis to plot on and which are returned.
        **kwargs: key arguments passed to plt.figure (if fig and ax are None).
        Other arguments: see plot3d_orbit.
        """
#         raise NotImplementedError('plot2d_orbit not implemented.')
        assert R>=0, 'Planet radius must be positive'
        
        f=np.linspace(0,360,res)
        ret_fig = (fig is not None or ax is not None)
        if fig is None and ax is None: fig = plt.figure(**kwargs)#,dpi=80)
        if ax is None: ax = fig.add_subplot(111)
        ax.add_patch(plt.Circle([0,0], radius=R, color=planet_color))
#         axis equal
        
        axis1,axis2={'x':(1,2),'y':(0,2),'z':(0,1)}[axis]
        
        minv,maxv=[-R,-R,-R],[R,R,R]
        for orb in orbs:
            if len(orb)==3:
                orb,minf,maxf = orb
            elif len(orb)==2:
                orb,maxf = orb
                minf = orb.f
            elif len(orb)==1:
                if isinstance(orb,list): orb=orb[0]
                minf=0
                maxf=360
            else: raise TypeError("Invalid input:",orb)
            if np.isclose(maxf%360,minf): minf,maxf=0,360
            else: maxf=(maxf-minf)%360+minf
            f=np.linspace(minf,maxf,res)
            R,_=space().calc_rv(*orb.coe()[:-1],f,orb.mu);
            minv=np.amin(np.vstack([R,minv]),axis=0)
            maxv=np.amax(np.vstack([R,maxv]),axis=0)
            ax.plot(R[:,axis1],R[:,axis2],linewidth=2.5)
            if (0-minf)%360<=(maxf-minf)%360 or (maxf-minf)%360==0:
                R,_=space().calc_rv(*orb.coe()[:-1],0,orb.mu);
                ax.plot(R[axis1],R[axis2],'s',ms=10,mew=2,mec='lime',mfc='white')
            if (180-minf)%360<=(maxf-minf)%360 or (maxf-minf)%360==0:
                R,_=space().calc_rv(*orb.coe()[:-1],180,orb.mu);
                ax.plot(R[axis1],R[axis2],marker='s',ms=10,mew=2,mec='r',mfc='white')
            R,V=orb.rv()
            V=V*vscale
            ax.plot(R[axis1],R[axis2],marker='o',ms=10,mew=2,mec='b',mfc=[0,0,0,0])
# #             ax.quiver3D(R[0],R[1],R[2],V[0],V[1],V[2],length=vscale)
            V=R+V
            minv=np.amin(np.vstack([V,minv]),axis=0)
            maxv=np.amax(np.vstack([V,maxv]),axis=0)
#             arrow = Arrow3D([R[0], V[0]], [R[1], V[1]], 
#                         [R[2], V[2]], mutation_scale=20, 
#                         lw=3, arrowstyle="-|>", color="r")
#             ax.add_artist(arrow)
            arrow = FancyArrowPatch([R[axis1],R[axis2]],[V[axis1],V[axis2]], mutation_scale=20, lw=3, arrowstyle="-|>",color="r")
            ax.add_patch(arrow)
#         axis equal
        ax.tick_params(which='both',direction='in',top=True,right=True,bottom=True,left=True)
        ax.minorticks_on()
#         ax.set_aspect(1)
#         ax.set_box_aspect(np.ptp(np.vstack([minv,maxv]),axis=0))
        if ret_fig: return minv,maxv
        ax.axis('equal')
        fig.tight_layout()

        plt.show()
        
    def plot3d_flat_orbit(self,orbs,R=6371,res=100,planet_color=[0,.25,0],vscale=1e3,save=None,**kwargs):
        """
        Plot the given orbits using 2D orthographic projection, which results in
        three subplots, one for each view axis.
        save: Name with file extension to save the resulting plot.
        Other arguments: See plot2d_orbit.
        """
        fig = plt.figure(**kwargs)
        gs = GridSpec(2, 2)
        axs = [fig.add_subplot(gs[2])]
        axs.append(fig.add_subplot(gs[3]))
        axs.append(fig.add_subplot(gs[0]))
        for ax,axis in zip(axs,['y','x','z']):
            minv,maxv=space().plot2d_orbit(orbs,ax=ax,axis=axis,R=R,res=res,planet_color=planet_color,vscale=vscale)
            ax.set_aspect('equal','datalim',share=True)
        gs.set_width_ratios([maxv[0]-minv[0],maxv[1]-minv[1]])
        gs.set_height_ratios([maxv[1]-minv[1],maxv[2]-minv[2]])
        axs[0].set_xlabel('x')
        axs[2].set_ylabel('y')
        axs[1].set_xlabel('y')
        axs[0].set_ylabel('z')
        plt.setp(axs[2].get_xticklabels(), visible=False)
        plt.setp(axs[1].get_yticklabels(), visible=False)
        fig.subplots_adjust(.1,.07,.99,.99,0,0)
        if save:
            fig.savefig(save)
            print('Saved:',save)
        plt.show()
        
    def R2f(self,orb,R,supp_warn=False):
        """
        Returns the true anomaly on the orbit where the position vector is in the
        same direction as the given vector. Gives a warning if the given vector
        is not in the orbital plane and the warning is not suppressed with
        supp_warn=True. Suppressing the warning should be avoided and only used
        when it is known for sure that there is a corresponding valid true anomaly,
        as it can otherwise give out an incorrect value.
        Note: The formula for finding f (true anomaly) can be used to find omega
        instead by exchanging f and omega.
        """
        if abs(np.cross(orb.R,orb.V).dot(R)/(norm(np.cross(orb.R,orb.V))*norm(R)))>np.finfo(np.float64).eps and not supp_warn:
            warnings.warn('Given position detected as not on the orbital plane. Suppress this warning with supp_warn=True')
#             raise Exception('Position is not on the orbital plane.')
#         else: print(np.cross(orb.R,orb.V).dot(R)/(norm(np.cross(orb.R,orb.V))*norm(R)))
        theta = np.arctan2(np.sqrt(R[0]**2+R[1]**2),R[2])
        phi = np.arctan2(R[1],R[0])
        omega = np.deg2rad(orb.omega)
        Omega = np.deg2rad(orb.Omega)
        i = np.deg2rad(orb.i)
        f = np.rad2deg(-omega-np.arctan2(np.sin(theta)*np.cos(Omega-phi),np.sin(i)*np.cos(theta)-np.sin(theta)*np.sin(Omega-phi)*np.cos(i)))+90 #Some magic
        return f%360
    
    def orb_plane_inters(self,o1,o2):
        """
        Returns the true anomaly on orbit o1 which is at the intersection of
        the orbital planes of o1 and o2 and where o2 appears to o1 as at its
        ascending node.
        
        o1, o2: Orbit instances
        """
        h1=np.cross(*o1.rv())
        h2=np.cross(*o2.rv())
        h3=np.cross(h1,h2)
        return space().R2f(o1,h3,supp_warn=True)
    
    def trans_orb_plane(self,o1,o2,new_orbit=True):
        """
        Returns a new orbit equivalent to rotating the orbit o1 around the
        intersection of the orbital planes of o1 and o2 for it to be on the same
        orbital plane as o2. For new_orbit=False, only returns the value omega
        for this new orbit.
        """
        h1=np.cross(*o1.rv())
        h2=np.cross(*o2.rv())
        h3=np.cross(h1,h2)
        orb_out = orbit(a=o1.a,e=o1.e,i=o2.i,omega=space().orb_plane_inters(o1,o2),Omega=o2.Omega,f=o1.f)
        omega = space().R2f(orb_out,h3,supp_warn=True) # Exchanging f and omega in the formula allows to find one or the other; see note under R2f.
        if new_orbit: return orb_out.update(omega=omega)
        return omega%360
#         theta = np.arctan2(np.sqrt(h3[0]**2+h3[1]**2),h3[2])
#         phi = np.arctan2(h3[1],h3[0])
# #         omega = np.deg2rad(o1.omega)
#         f = np.deg2rad(space().orb_plane_inters(o1,o2))
#         Omega = np.deg2rad(o2.Omega)
#         i = np.deg2rad(o2.i)
#         omega = np.rad2deg(-f-np.arctan2(np.sin(theta)*np.cos(Omega-phi),np.sin(i)*np.cos(theta)-np.sin(theta)*np.sin(Omega-phi)*np.cos(i)))+90 #Some magic
#         if new_orbit: return orbit(a=o1.a,e=o1.e,i=o2.i,omega=omega,Omega=o2.Omega,f=o1.f)
#         return omega%360

    def _R(self,phi,o1,o2):
        """
        Difference in distance between the two object in direction phi.
        """
        return o1.a*(1-o1.e**2)/(1+o1.e*np.cos(np.deg2rad(phi-o1.omega)))-o2.a*(1-o2.e**2)/(1+o2.e*np.cos(np.deg2rad(phi-o2.omega)))

    def _dR(self,phi,o1,o2):
        """
        d _R/d phi
        """
        return o1.a*o1.e*(1-o1.e**2)*np.sin(np.deg2rad(phi-o1.omega))/(1+o1.e*np.cos(np.deg2rad(phi-o1.omega)))**2-o2.a*o2.e*(1-o2.e**2)*np.sin(np.deg2rad(phi-o2.omega))/(1+o2.e*np.cos(np.deg2rad(phi-o2.omega)))**2
    
#     def _ddR(phi,o1,o2):
#         return 2*o1.a*o1.e**2*(1-o1.e**2)*np.sin(np.deg2rad(phi-o1.omega))**2/(o1.e*np.cos(np.deg2rad(phi-o1.omega))+1)**3 
# + o1.a*o1.e*(1 - o1.e**2)*np.cos(np.deg2rad(phi-o1.omega))/(o1.e*np.cos(np.deg2rad(phi-o1.omega))+1)**2 
# - 2*o2.a*o2.e**2*(1-o2.e**2)*np.sin(np.deg2rad(phi-o2.omega))**2/(o2.e*np.cos(np.deg2rad(phi-o2.omega))+1)**3 
# - o2.a*o2.e*(1-o2.e**2)*np.cos(np.deg2rad(phi-o2.omega))/(o2.e*np.cos(np.deg2rad(phi-o2.omega))+1)**2
    
    def orb_inters(self,o1,o2,guess=0): #Note: return phi (f1=phi-o1.omega)
        """
        Intersection between two orbits (regardless of orbital plane).
        Returns phi: phi = f1 + o1.omega = f2 + o2.omega
        """
        phi = fsolve(space()._R,guess,(o1,o2))[0]
        if np.isclose(norm(o1.copy().update(f=phi-o1.omega).R),norm(o2.copy().update(f=phi-o2.omega).R)): return phi%360
        raise RuntimeError("No intersection found.")
    
    def orb_paral(self,o1,o2,guess=0): #Note: return phi (f1=phi-o1.omega)
        """
        Where the orbits are parallel to each other.
        Returns phi: phi = f1 + o1.omega = f2 + o2.omega
        """
        return fsolve(space()._dR,guess,(o1,o2))[0]%360
    
    def _diff_int_par(self,o1,o2,guess=0):
        """
        Difference in degrees between intersection of orbits and parallel point.
        """
        try:
            phi1=space().orb_inters(o1,o2,guess)
            phi2=space().orb_paral(o1,o2,o1.f+o1.omega+180)
            return 180-abs(180-abs(phi1-phi2)%360)
        except (RuntimeError,TypeError): # It's ok to use Exception instead of RuntimeError to ignore actual runtime error from scipy.optimize.fsolve
            return 180

    def _impulse_func(self,scale,o1,o2,guess=0):
        """
        Multiply velocity of orbit o1 and _diff_int_par using this new orbit.
        """
        return space()._diff_int_par(o1.copy().update(V=o1.V*scale),o2,guess)
    
    def scale_burn(self,o1,o2,guess=None):
        """
        How much to multiply the velocity of orbit o1 to make it tangential to
        orbit o2.
        """
        if guess is None:
            if o1.a>o2.a:
                guess = (o1.mu*(2/norm(o1.R)-2/(o1.ap+o2.pe)))**(1/2)/norm(o1.V)
            else:
                guess = (o1.mu*(2/norm(o1.R)-2/(o1.pe+o2.ap)))**(1/2)/norm(o1.V)
        return abs(fsolve(space()._impulse_func,guess,(o1,o2))[0])
    
    def phasing_orbits(self,orb,f,n=1,minR=6371+200,asap=False):
        """
        Returns the necessary delta V at current orbit position of orbit orb in
        order to make a rendezvous with an object on the same orbit at true
        anomaly f.
        n: Number of revolutions to do before rendezvous (reduces delta V).
        minR: Minimum altitude permitted.
        asap: If true, favors the fastest option (n should be 1 to really be fastest).
        """
        o1 = orb.copy()
        o2 = orb.copy().update(f=f)
        phi = (o2.M-o1.M)%360/n
        phi1 = 360-phi
        phi2 = phi1+360
        a1 = o1.a*(phi1/360)**(2/3)
        dV1 = (o1.mu*(2/norm(o1.R)-1/a1))**(1/2)-norm(o1.V)
        a2 = o1.a*(phi2/360)**(2/3)
        dV2 = (o1.mu*(2/norm(o1.R)-1/a2))**(1/2)-norm(o1.V)
        if o1.copy().dV(dV1).pe < minR: return dV2
        if asap or abs(dV1)<abs(dV2): return dV1
        return dV2
    
#####################################################
    def orbit_transfer(self,o1,o2,direct_incl=False,phasing=False,n=1,minR=6371+100,asap=False,output=None,full_output=False):
        """
        Calculates the transfer between two orbits.
        By default, calculates to minimize the delta V. Note that increasing n
        can help minimize the delta V.
        
        o1, o2: Initial orbit, target orbit
        direct_incl: Affects the delta V for inclination change:
            False: Returns a delta V which needs to be applied normally
            northwardly (+) or southwardly (-) to the current orbit.
            True: Returns a net delta V to pass directly from the initial orbit
            to the orbit on the orbit aligned with the target orbit.
        phasing: If True, calculates a phasing orbit (for rendezvous with target).
        n, minR: Passed to phasing_orbits() method.
        asap: If true, calculates to minimize the transfer time. Otherwise,
        calculates towards minimizing delta V.
        output: pandas.DataFrame to output the data. Data appended inplace.
        full_output: If true, returns the full output dataframe, a list of index
        for the dataframe, the total delta V, and the distance from target it
        reaches right after the second impulse for the phasing orbit.
        """
        o1=o1.copy()
        o2=o2.copy()
        if output is None: output = pd.DataFrame(columns=['step','f','time','deltaV','orbit'])
        else: o1=output['orbit'].iloc[-1]
        deltaV_tot=0
        useful=[0]
        idmin=0
        dist=None

        ## Initial
        output.loc[0]=['Initial',o1.f,'00:00:00',0,o1]
        ## Inclination
        if not all(np.isclose([o1.i,o1.Omega],[o2.i,o2.Omega])): # No inclination calculations if already on same plane.
            ## Inclination
            f1=space().orb_plane_inters(o1,o2)
            f2=(f1+180)%360
            if (f1-o1.f)%360<(f2-o1.f)%360:
                output = output.append({'step':'Incl nml+'},ignore_index=True)
                output = output.append({'step':'Incl nml-'},ignore_index=True)
            else:
                output = output.append({'step':'Incl nml-'},ignore_index=True)
                output = output.append({'step':'Incl nml+'},ignore_index=True)
                f1,f2=f2,f1
            output['f'].iloc[[-2,-1]]=[f1,f2]
            output['time'].iloc[-2]="%02d:%02d:%05.2f" %s2hms((o1.copy().update(f=f1).M-o1.M)%360/360*o1.P+hms2s(*list(map(float,output['time'].loc[idmin].split(':')))))
            output['time'].iloc[-1]="%02d:%02d:%05.2f" %s2hms((o1.copy().update(f=f2).M-o1.M)%360/360*o1.P+hms2s(*list(map(float,output['time'].loc[idmin].split(':')))))
            o3 = space().trans_orb_plane(o1,o2).update(f=f1)
            if direct_incl: output['deltaV'].iloc[-2]=o3.V-o1.copy().update(f=f1).V
            else: output['deltaV'].iloc[-2]=norm(o3.V)*ang_bet(o3.V,o1.copy().update(f=f1).V)
            output['orbit'].iloc[-2]=o3.copy()

            o3.update(f=f2)
            if direct_incl: output['deltaV'].iloc[-1]=o3.V-o1.copy().update(f=f2).V
            else: output['deltaV'].iloc[-1]=norm(o3.V)*ang_bet(o3.V,o1.copy().update(f=f2).V)
            output['orbit'].iloc[-1]=o3.copy()

            if asap:
                idmax = output.index[-1]
                idmin = output.index[-2]
            else:
                idmax = output.loc[output['step'].str.contains('Incl'),'deltaV'].map(norm).idxmax()
                idmin = output.loc[output['step'].str.contains('Incl'),'deltaV'].map(norm).idxmin()
            output.loc[idmax,'time']="("+output.loc[idmax,'time']+")"
            o1=o3
            o1.update(f=output.loc[idmin,'f'])

            deltaV_tot+=norm(output.loc[idmin,'deltaV'])
            useful.append(idmin)

        if not all(np.isclose([o1.a,o1.e,o1.omega],[o2.a,o2.e,o2.omega])): # No further orbit transfer if already the same.
            try:
                with warnings.catch_warnings(record=True):
                    phi1=space().orb_inters(o1,o2,0)
                    phi2=space().orb_paral(o1,o2,phi1)
                    if not np.isclose(phi1,phi2): phi3=space().orb_inters(o1,o2,phi2*2-phi1)
                    else: phi3=phi1
                    phi4=space().orb_paral(o1,o2,phi2+180)
#         #         display(phi1,phi2,phi3,phi4)
            except RuntimeError:
                asap=False
                phi2=space().orb_paral(o1,o2,o1.f)
                phi4=space().orb_paral(o1,o2,phi2+180)
#         #         display(phi2,phi4)
            if asap:
                output = output.append({'step':'Intersection 1'},ignore_index=True)
                output = output.append({'step':'Intersection 2'},ignore_index=True)
                if (phi1-o1.omega-o1.f)%360>(phi3-o1.omega-o1.f)%360: phi1,phi3=phi3,phi1
                o3 = output['orbit'].loc[idmin].copy()
                output['f'].iloc[-2]=o3.update(f=(phi1-o1.omega)%360).f
                output['time'].iloc[-2]="%02d:%02d:%05.2f" %s2hms((o3.M-o1.M)%360/360*o1.P+hms2s(*list(map(float,output['time'].loc[idmin].split(':')))))
                output['orbit'].iloc[-2]=o2.copy().update(f=(phi1-o2.omega)%360)
                output['deltaV'].iloc[-2]=output['orbit'].iloc[-2].V-o3.V
                deltaV_tot+=norm(output['deltaV'].iloc[-2])

                o3 = output['orbit'].loc[idmin].copy()
                output['f'].iloc[-1]=o3.update(f=(phi2-o1.omega)%360).f
                output['time'].iloc[-1]="(%02d:%02d:%05.2f)" %s2hms((o3.M-o1.M)%360/360*o1.P+hms2s(*list(map(float,output['time'].loc[idmin].split(':')))))
                output['orbit'].iloc[-1]=o2.copy().update(f=(phi2-o2.omega)%360)
                output['deltaV'].iloc[-1]=output['orbit'].iloc[-1].V-o3.V

                idmax = output.index[-1]
                idmin = output.index[-2]
                useful.append(idmin)

            else:
                ### Option 1
                option1 = pd.DataFrame(columns=['step','f','time','deltaV','orbit'])
                option1 = option1.append({'step':'Impulse 1'},ignore_index=True)
                option1 = option1.append({'step':'Impulse 2'},ignore_index=True)
                o3 = output['orbit'].loc[idmin].copy()
                option1.loc[0,'f']=(phi2-o3.omega)%360
                option1.loc[0,'time']="%02d:%02d:%05.2f" %s2hms((o3.copy().update(f=(phi2-o3.omega)%360).M-o3.M)%360/360*o3.P+hms2s(*list(map(float,output['time'].loc[idmin].split(':')))))
                with warnings.catch_warnings(record=True): #For restoring warnings filters upon exit.
#                     warnings.simplefilter("ignore",category="RuntimeWarning")
                    scale = space().scale_burn(o3.update(f=(phi2-o3.omega)%360),o2)
                option1.loc[0,'deltaV']=(scale-1)*norm(o3.V)
                option1.loc[0,'orbit']=o3.update(V=o3.V*scale).copy()
                
                try:
                    phi5=space().orb_inters(o3,o2,phi2+180)
                except RuntimeError:
                    phi5=space().orb_paral(o3,o2,phi2+180)
                option1.loc[1,'f']=(phi5-o3.omega)%360
                option1.loc[1,'time']="%02d:%02d:%05.2f" %s2hms((o3.copy().update(f=(phi5-o3.omega)%360).M-o3.M)%360/360*o3.P+hms2s(*list(map(float,option1['time'].loc[0].split(':')))))
                o3.update(f=(phi5-o3.omega)%360);
                option1.loc[1,'deltaV']=(norm(o2.copy().update(f=phi5-o2.omega).V)/norm(o3.V)-1)*norm(o3.V)
                option1.loc[1,'orbit']=o3.dV(option1.loc[1,'deltaV']).copy()

                deltaV_option1=sum(option1['deltaV'].map(norm))

#         #         display(option1)

                ### Option 2
                option2 = pd.DataFrame(columns=['step','f','time','deltaV','orbit'])
                option2 = option2.append({'step':'Impulse 1'},ignore_index=True)
                option2 = option2.append({'step':'Impulse 2'},ignore_index=True)
                o3 = output['orbit'].loc[idmin].copy()
                option2.loc[0,'f']=(phi4-o3.omega)%360
                option2.loc[0,'time']="%02d:%02d:%05.2f" %s2hms((o3.copy().update(f=(phi4-o3.omega)%360).M-o3.M)%360/360*o3.P+hms2s(*list(map(float,output['time'].loc[idmin].split(':')))))
                with warnings.catch_warnings(record=True):
#                     warnings.simplefilter("ignore",category="RuntimeWarning")
                    scale = space().scale_burn(o3.update(f=(phi4-o3.omega)%360),o2)
                option2.loc[0,'deltaV']=(scale-1)*norm(o3.V)
                option2.loc[0,'orbit']=o3.update(V=o3.V*scale).copy()
                
                try:
                    phi6=space().orb_inters(o3,o2,phi4+180)
                except RuntimeError:
                    phi6=space().orb_paral(o3,o2,phi4+180)
                option2.loc[1,'f']=(phi6-o3.omega)%360
                option2.loc[1,'time']="%02d:%02d:%05.2f" %s2hms((o3.copy().update(f=(phi6-o3.omega)%360).M-o3.M)%360/360*o3.P+hms2s(*list(map(float,option2['time'].loc[0].split(':')))))
                o3.update(f=(phi6-o3.omega)%360);
                option2.loc[1,'deltaV']=(norm(o2.copy().update(f=phi6-o2.omega).V)/norm(o3.V)-1)*norm(o3.V)
                option2.loc[1,'orbit']=o3.dV(option2.loc[1,'deltaV']).copy()

                deltaV_option2=sum(option2['deltaV'].map(norm))

#         #         display(option2)

                ### Choosing best option and append
                if deltaV_option1<deltaV_option2:
                    output = pd.concat([output,option1,option2],ignore_index=True)
                    deltaV_tot+=deltaV_option1
                else:
                    output = pd.concat([output,option2,option1],ignore_index=True)
                    deltaV_tot+=deltaV_option2

                output['step'].iloc[-4:-2]="Option 1: "+output['step'].iloc[-4:-2]
                output['step'].iloc[-2:]="Option 2: "+output['step'].iloc[-2:]
                output['time'].iloc[-2:]="("+output['time'].iloc[-2:]+")"
                idmin = output.index[-3]
                useful.append(output.index[-4])
                useful.append(idmin)

        if phasing and not np.isclose(o1.f,o2.f):
            output = output.append({'step':'Phasing 1'},ignore_index=True)
            o3 = output['orbit'].loc[idmin].copy()
            if asap:
                output['f'].iloc[-1]=o3.f
                output['time'].iloc[-1]=output['time'].loc[idmin]
            else:
                output['f'].iloc[-1]=0 # More efficient when smaller R
                output['time'].iloc[-1]="%02d:%02d:%05.2f" %s2hms((0-o3.M)%360/360*o3.P+hms2s(*list(map(float,output['time'].loc[idmin].split(':')))))
                o3.update(f=0);
            time = hms2s(*list(map(float,output['time'].iloc[-1].split(':'))))
            f = space().predict(o2,time,rv=False)+o2.omega-o3.omega
#         #     print(f)
            output['deltaV'].iloc[-1]=space().phasing_orbits(o3,f,n=n,minR=minR,asap=asap)
            output['orbit'].iloc[-1]=o3.dV(output['deltaV'].iloc[-1]).copy()
#         #     print(o3.a)

            deltaV_tot+=norm(output['deltaV'].iloc[-1])
            useful.append(output.index[-1])

            output = output.append({'step':'Phasing 2'},ignore_index=True)
            output['f'].iloc[-1]=o3.f
            output['time'].iloc[-1]="%02d:%02d:%05.2f" %s2hms(o3.P*n+hms2s(*list(map(float,output['time'].iloc[-2].split(':')))))
            output['deltaV'].iloc[-1]=-output['deltaV'].iloc[-2]
            output['orbit'].iloc[-1]=o3.dV(output['deltaV'].iloc[-1]).copy()
#         #     print(o3.a)

            deltaV_tot+=norm(output['deltaV'].iloc[-1])
            idmin=output.index[-1]
            useful.append(idmin)
            
            time = hms2s(*list(map(float,output['time'].iloc[-1].split(':'))))
            f = space().predict(o2,time,rv=False)
            dist=norm(o3.copy().R-o2.copy().update(f=f).R)

        output = output.append({'step':'Final','f':o2.f,'deltaV':0},ignore_index=True)
        o3 = output['orbit'].loc[idmin]
        output['orbit'].iloc[-1]=o3.copy().update(f=o2.f+o2.omega-o3.omega)
        output['time'].iloc[-1] = "%02d:%02d:%05.2f" %s2hms((output['orbit'].iloc[-1].M-output['orbit'].loc[idmin].M)%360/360*o2.P+hms2s(*list(map(float,output['time'].loc[idmin].split(':')))))

        useful.append(output.index[-1])

#         # if output['orbit'].iloc[-1]!=o2: print("Orbit transfer calculations failed.")
#         # else: print("Success!")

        if full_output: return output,useful,deltaV_tot,dist
        else: 
            output = output.loc[useful].reset_index(drop=True)
            output.loc[output['step'].str.contains('Option 1: '),'step'] = output.loc[output['step'].str.contains('Option 1: '),'step'].map(lambda x: x.removeprefix("Option 1: "))
            output.loc[output['step'].str.contains('Option 2: '),'step'] = output.loc[output['step'].str.contains('Option 2: '),'step'].map(lambda x: x.removeprefix("Option 2: "))
            return output
    
    def plot_orbit_transfer(self,data,plot=True,dim=3,**kwargs):
        """
        Plots the data returned by orbit_transfer() (with full_output=False,
        default), or returns a list ready to be plotted.
        
        data: pandas.DataFrame as returned by orbit_transfer() with full_output=False
        plot: If true, plots the data. Otherwise, returns a list ready to be
        passed to plot3d_orbit, plot2d_orbit or plot3d_flat_orbit.
        dim: {2,3} If 2, plots in 2D with plot3d_flat_orbit(). If 3, plots in 3D
        with plot3d_orbit().
        **kwargs: Are passed to plot3d_orbit or plot3d_flat_orbit.
        """
        list_transfer=data[['orbit','f']].to_numpy().tolist()
        list_transfer[0]=list_transfer[0][0]
        list_transfer.insert(1,[list_transfer[0]])
        list_transfer[1:-1]=list(map(lambda x,y: [x[0],y[1]],list_transfer[1:-1],list_transfer[2:]))
        list_transfer[-1]=list_transfer[-1][0]
        if plot:
            if dim==3: space().plot3d_orbit(list_transfer,**kwargs)
            if dim==2: space().plot3d_flat_orbit(list_transfer,**kwargs)
        else: return list_transfer


#############################################################################

"""
Defines a mercator scale which is used for plotting the ground track.
From https://matplotlib.org/stable/gallery/scales/custom_scale.html
"""

from numpy import ma
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
# %matplotlib inline

class MercatorLatitudeScale(mscale.ScaleBase):
    name = 'mercator'

    def __init__(self, axis, *, thresh=np.deg2rad(85), **kwargs):
        super().__init__(axis)
        if thresh >= np.pi / 2:
            raise ValueError("thresh must be less than pi/2")
        self.thresh = thresh

    def get_transform(self):
        return self.MercatorLatitudeTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):
        fmt = FuncFormatter(
            lambda x, pos=None: f"{np.degrees(x):.0f}\N{DEGREE SIGN}")
        axis.set(major_locator=FixedLocator(np.radians(range(-90, 90, 15))),
                 major_formatter=fmt, minor_formatter=fmt)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, -self.thresh), min(vmax, self.thresh)

    class MercatorLatitudeTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)
            if masked.mask.any():
                return ma.log(np.abs(ma.tan(masked) + 1 / ma.cos(masked)))
            else:
                return np.log(np.abs(np.tan(a) + 1 / np.cos(a)))

        def inverted(self):
            return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(
                self.thresh)

    class InvertedMercatorLatitudeTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            return np.arctan(np.sinh(a))

        def inverted(self):
            return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)
        
mscale.register_scale(MercatorLatitudeScale)

############################################################################

"""
Defines a (good looking) 3D arrow for matplotlib
From https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
"""

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
