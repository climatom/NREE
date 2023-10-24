#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boiler-plate functions for fast humid heat computations

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORTS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import numba as nb
from numba import prange
from netCDF4 import Dataset
import sys 
import math
import time


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

eps=0.05 # Tw toleance (C)
itermax=10 # max iterations for Tw convergence
rh_thresh=1. # %. Below this, we assume RH is this. 
    
@nb.njit(fastmath=True)
def _satVP(t):
    """Saturation specific humidity from temp (k)"""
    # t=np.atleast_1d(t)
    # esat=np.zeros(t.shape)
    t0=273.16
    a1_w=611.21; a3_w=17.502; a4_w=32.19
    a1_i=611.21; a3_i=22.587; a4_i=-0.7
    
    # Sat Vp according to Teten's formula
    if t>273.15:
        esat=a1_w*np.exp(a3_w*(t-t0)/(t-a4_w))
    else:
        esat=a1_i*np.exp(a3_i*(t-t0)/(t-a4_i))   
        
    return esat

@nb.jit("float64[:](float64[:])")
def satvp_huang(tc):
    vp=np.ones(len(tc))*np.nan
    vp[tc>0]=np.exp(34.494-4924.99/(tc[tc>0]+237.1))/(tc[tc>0]+105)**1.57
    vp[tc<=0]=np.exp(43.494-6545.8/(tc[tc<=0]+278.))/(tc[tc<=0]+868)**2
    return vp

@nb.jit
def satvp_huang_scalar(tc):
    """ Note that tc should be air temperature in degrees celsius!"""
    if tc>0: vp = np.exp(34.494-4924.99/(tc+237.1))/(tc+105.)**1.57
    else: vp = np.exp(43.494-6545.8/(tc+278.))/(tc+868.)**2
    return vp

@nb.njit(fastmath={"nnan":False},parallel=True)
def _satVp2D(t,nr,nc):
    """Saturation specific humidity from 2d array of temp (k)"""
    t0=273.16
    a1_w=611.21; a3_w=17.502; a4_w=32.19
    a1_i=611.21; a3_i=22.587; a4_i=-0.7
    esat=np.zeros((nr,nc))
    for i in prange(nr):
        for j in prange(nc):
            if np.isnan(t[i,j]): continue
            if t[i,j]>273.15:
                esat[i,j]=a1_w*np.exp(a3_w*(t[i,j]-t0)/(t[i,j]-a4_w))
            else:
                esat[i,j]=a1_i*np.exp(a3_i*(t[i,j]-t0)/(t[i,j]-a4_i))
    return esat    

@nb.njit(fastmath={"nnan":False},parallel=True)
def _satVP3D(nt,nr,nc,t):    
    out=np.zeros((nt,nr,nc))*np.nan
    for _t in prange(nt):
        for _r in prange(nr):
            for _c in prange(nc):
                if np.isnan(t[_t,_r,_c]): continue
                out[_t,_r,_c]=_satVP(t[_t,_r,_c])
    return out

@nb.njit(fastmath={"nnan":False},parallel=True)
def _dew_rh_3d(nt,nr,nc,t,td):    
    """ Note that input must be in deg C!"""
    out=np.zeros((nt,nr,nc))*np.nan
    for _t in prange(nt):
        for _r in prange(nr):
            for _c in prange(nc):
                if np.isnan(t[_t,_r,_c]): continue
                if np.isnan(td[_t,_r,_c]): continue
                s1=satvp_huang(t[_t,_r,_c])
                s2=satvp_huang(td[_t,_r,_c])
                rh=s2/s1
                if rh>1: rh=1.
                if rh<0: rh=0.
                out[_t,_r,_c]=rh
              
    return out

@nb.njit(fastmath={"nnan":False},parallel=True)
def _satvp_huang_3d(nt,nr,nc,t):    
    """ Note that input must be in deg C!"""
    out=np.zeros((nt,nr,nc))*np.nan
    for _t in prange(nt):
        for _r in prange(nr):
            for _c in prange(nc):
                if np.isnan(t[_t,_r,_c]): continue
                out[_t,_r,_c]=satvp_huang(t[_t,_r,_c])
    return out


@nb.jit
def _dewRH3D(nt,nr,nc,t,td):    
    """ takes T and Td in K"""
    out=np.zeros((nt,nr,nc))*np.nan
    for _t in prange(nt):
        for _r in prange(nr):
            for _c in prange(nc):
                if np.isnan(t[_t,_r,_c]): continue
                out[_t,_r,_c]=satvp_huang_scalar\
                    (td[_t,_r,_c]-273.15)/satvp_huang_scalar(t[_t,_r,_c]-273.15)
                if out[_t,_r,_c]>1.: out[_t,_r,_c]=1.
    return out

@nb.njit(fastmath=True)
def _dewRH2D(nr,nc,t,td):    
    """ takes T and Td in K"""
    out=np.zeros((nr,nc))*np.nan
    for _r in prange(nr):
        for _c in prange(nc):
                if np.isnan(t[_r,_c]): continue
                out[_r,_c]=satvp_huang_scalar\
                    (td[_r,_c]-273.15)/satvp_huang_scalar(t[_r,_c]-273.15)
                if out[_r,_c]>1.: out[_r,_c]=1.
    return out



@nb.njit(fastmath=True)
def _rhDew(rh,t):
   """ Dewpoint from t (K) and rh (%)"""
   a1=611.21; a3=17.502; a4=32.19
   t0=273.16
   vp=satvp_huang_scalar(t-273.15)*rh/100.
   dew=(a3*t0 - a4*np.log(vp/a1))/(a3 - np.log(vp/a1))
   return dew

def _rhDew2d(nr,nc,rh,t):
   """ Dewpoint from t (K) and rh (%)"""
   a1=611.21; a3=17.502; a4=32.19
   t0=273.16
   dew=np.zeros((nr,nc))
   for i in prange(nr):
       for j in prange(nc):
           vp=satvp_huang_scalar(t[i,j]-273.15)*rh[i,j]/100.
           dew[i,j]=(a3*t0 - a4*np.log(vp/a1))/(a3 - np.log(vp/a1))
   return dew

@nb.njit(fastmath=True)
def _satQ(t,p):
    """Saturation specific humidity from temp (k) and pressure (Pa)"""
    
    # t=np.atleast_1d(t)
    # esat=np.zeros(t.shape)
    t0=273.16
    a1_w=611.21; a3_w=17.502; a4_w=32.19
    a1_i=611.21; a3_i=22.587; a4_i=-0.7
    rat=0.622;
    rat_rec=1-rat
    # widx=t>273.1
    
    # Sat Vp according to Teten's formula
    if t>273.15:
        esat=a1_w*np.exp(a3_w*(t-t0)/(t-a4_w))
    else:
        esat=a1_i*np.exp(a3_i*(t-t0)/(t-a4_i))    
    # Now sat q according to Eq. 7.5 in ECMWF thermodynamics guide
    satQ=rat*esat/(p-rat_rec*esat)
    return satQ

@nb.njit(fastmath=True)
def _satQ_huang(t,p):
    
    """Saturation specific humidity from temp (k) and pressure (Pa)"""
    
    # t=np.atleast_1d(t)
    # esat=np.zeros(t.shape)
    rat=0.622;
    rat_rec=1-rat
    # widx=t>273.1
    
    tc=t-273.15
    if tc>0: esat = np.exp(34.494-4924.99/(tc+237.1))/(tc+105.)**1.57
    else: esat = np.exp(43.494-6545.8/(tc+278.))/(tc+868.)**2
    # Now sat q according to Eq. 7.5 in ECMWF thermodynamics guide
    satQ=rat*esat/(p-rat_rec*esat)
    return satQ


@nb.njit(fastmath={"nnan":False},parallel=True)
def _satQ3d(t,p,nt,nr,nc):
    out=np.zeros((nt,nr,nc))*np.nan
    for _t in nb.prange(nt):
        for _r in nb.prange(nr):
            for _c in nb.prange(nc):
                if np.isnan(t[_t,_r,_c]) or np.isnan(p[_t,_r,_c]): continue
                out[_t,_r,_c]=_satQ(t[_t,_r,_c],p[_t,_r,_c])
    return out

@nb.njit(fastmath=True)
def _LvCp(t,q):
    """ Latent heat of evaporation (J) and specific heat capacity of moist 
    air (J) from temp (K) and specific humidity (g/g)"""
    r=q/(1-q)
    Cp=1005.7*(1-r)+1850.* r*(1-r)
    Lv=1.918*1000000*np.power(t/(t-33.91),2)

    return Lv,Cp

@nb.njit(fastmath=True)
def _IMP(Tw,t,td,p):
    q=_satQ_huang(td,p)
    qs=_satQ_huang(Tw,p)
    Lv,Cp=_LvCp(t,q)
    diff=(t-Tw)-Lv/Cp*(qs-q)
    
    return diff

@nb.njit(fastmath={"nnan":False},parallel=True)
def _ME2D(nr,nc,t,td,p):
    me=np.zeros(td.shape)*np.nan
    for i in prange(nr):
        for j in prange(nc):   
            if np.isnan(t[i,j]) or np.isnan(td[i,j]) or np.isnan(p[i,j]): continue
            q=_satQ_huang(td[i,j],p[i,j])
            Lv,Cp=_LvCp(t[i,j],q)
            me[i,j]=Cp*t[i,j]+q*Lv            
    return me

@nb.njit(fastmath={"nnan":False},parallel=True)
def _ME3D(nt,nr,nc,t,td,p):
    """ Computes the moist enthalpy from temp and dewpoint temp; both in K"""
    me=np.zeros(td.shape)*np.nan
    q=np.zeros(td.shape)*np.nan
    Lv=np.zeros(td.shape)*np.nan
    Cp=np.zeros(td.shape)*np.nan
    for ti in prange(nt):
        for i in prange(nr):
            for j in prange(nc):   
                if np.isnan(t[ti,i,j]) or np.isnan(td[ti,i,j]) or \
                    np.isnan(p[ti,i,j]): continue
                
                q[ti,i,j]=_satQ(td[ti,i,j],p[ti,i,j])
                if np.isnan(q[ti,i,j]): print(td[ti,i,j],p[ti,i,j])
                Lv[ti,i,j],Cp[ti,i,j]=_LvCp(t[ti,i,j],q[ti,i,j])
                me[ti,i,j]=Cp[ti,i,j]*t[ti,i,j]+q[ti,i,j]*Lv[ti,i,j]           
    return me

@nb.njit(fastmath={"nnan":False},parallel=True)
def _ME1D(nt,t,rh,p):
    me=np.zeros(t.shape)*np.nan
    for i in prange(nt): 
            if np.isnan(t[i]) or np.isnan(rh[i]) or np.isnan(p[i]): continue
            td=_rhDew(rh[i],t[i])
            q=_satQ_huang(td,p[i])
            Lv,Cp=_LvCp(t[i],q)
            me[i]=Cp*t[i]+q*Lv            
    return me


@nb.njit(fastmath={"nnan":False})
def _TWStull(ta,rh):
    """ Computes Tw as a function of temperature (C) and RH (%)
    Ref: Stull (2011) 
    https://tinyurl.com/7k9xarrs
    """
    
    tw=ta*np.arctan(0.151977*(rh+8.313659)**0.5) + np.arctan(ta+rh) - \
    np.arctan(rh-1.676331)+0.00391838*(rh)**1.5*np.arctan(0.023101*rh)\
    - 4.686035
    
    return tw

@nb.njit(fastmath={"nnan":False},parallel=True)
def _TW(nt,ta,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        c p T + Lq = c p TW + εLe sat (TW)/p 
        
    Using the Newton-Rhapson method
        
    Note that t and td are in K; Tw is returned in K
    
    Note that this is the one-dimensional version of
    the below
    
    RH is in %
        
    """
    tw=np.zeros(nt,dtype=np.float32)*np.nan
    for _t in prange(nt):
                ni=0 # start iteration value
                # Protect against "zero" rh
                if rh[_t]<rh_thresh: td=_rhDew(rh_thresh,ta[_t])    
                else:td=_rhDew(rh[_t],ta[_t])
                # Initial guess. Assume saturation. 
                x0=ta[_t]-0. 
                f0=_IMP(x0,ta[_t],td,p[_t])
                if np.abs(f0)<=eps: tw[_t]=x0; continue # Got it first go!
                # Second guess, assume Tw=Ta-2    
                xi=x0-2.;
                fi=_IMP(xi,ta[_t],td,p[_t])
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx # first gradient
                if np.abs(fi)<=eps: tw[_t]=xi; continue # Got it 2nd go
                while np.abs(fi)>eps and ni<itermax:
                    xi=x0-f0/dfdx # new guess at Tw
                    fi=_IMP\
                        (xi,ta[_t],td,p[_t]) # error from this guess
                    if np.abs(fi)<=eps: tw[_t]=xi; break # Exit if small error
                    dx=(xi-x0)
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter

                # Catch odd behaviour of iteration 
                # If it didn't converge, set to nan    
                if ni == itermax: 
                    xi=np.nan
            
                # If it did converge, but reached a value >ta, set to ta 
                # [can happen when close to saturation and precision accepts
                # solution >ta] 
                elif xi >ta[_t,]:    
                    xi=ta[_t]*1. # V. close to saturation, so set to ta
    return tw

@nb.njit(fastmath={"nnan":False},parallel=True)  
def _TW2D(nr,nc,ta,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        c p T + Lq = c p TW + εLe sat (TW)/p 
        
    Using the Newton-Rhapson method
        
    Note that t is in K; rh is in %; p is in... and Tw is returned in K
    
        
    """
    tw=np.zeros((nr,nc),dtype=np.float32)*np.nan
    itermax=10 # max iterations
    for _r in range(nr):
        for _c in range(nc):
                ni=0 # start iteration value 
                if np.isnan(ta[_r,_c]) or np.isnan(rh[_r,_c]): continue
                td=_rhDew(rh[_r,_c],ta[_r,_c])
                # Initial guess. Assume saturation. 
                x0=ta[_r,_c]-0. 
                f0=_IMP(x0,ta[_r,_c],td,p[_r,_c])
                if np.abs(f0)<=eps: tw[_r,_c]=x0; continue # Got it first go!
                # Second guess, assume Tw=Ta-2    
                xi=x0-2.;
                fi=_IMP(xi,ta[_r,_c],td,p[_r,_c])
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx # first gradient
                if np.abs(fi)<=eps: tw[_r,_c]=xi; continue # Got it 2nd go
                while np.abs(fi)>eps and ni<itermax:
                    xi=x0-f0/dfdx # new guess at Tw
                    fi=_IMP\
                        (xi,ta[_r,_c],td,p[_r,_c]) # error from this guess
                    if np.abs(fi)<=eps: tw[_r,_c]=xi; break # Exit if small error
                    dx=(xi-x0)
                    # if np.abs(dx) <=eps: tw[_t,_r,_c]=xi; break # Exit if only need small change
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter

                # If it didn't converge, set to nan    
                if ni == itermax: 
                    tw[_r,_c]=np.nan
            
                # If it did converge, but reached a value >ta, set to ta 
                # [can happen when close to saturation and precision accepts
                # solution >ta] 
                elif xi >ta[_r,_c]:    
                    tw[_r,_c]=ta[_r,_c]*1. # V. close to saturation, so set to ta


    return tw


@nb.njit(fastmath={"nnan":False},parallel=True)  
def _TW3D(nt,nr,nc,ta,td,p):
    
    """ 
    Minimizes the implicit equation:
        
        cp*T + Lq = cp*TW + Le*sat(TW)/p 
        
    Using the Newton-Rhapson method
        
    Note that t is in K and q is g/g
    
    Tw is returned in K
        
    """
    
    tw=np.zeros((nt,nr,nc),dtype=np.float32)*np.nan

    for _r in range(nr):
        
        for _c in range(nc):           
        
            for _t in range(nt):
                
                if np.isnan(ta[_t,_r,_c]) or np.isnan(td[_t,_r,_c]) or np.isnan(p[_t,_r,_c]): 
                    continue
                            
                ni=0 # start iteration value
                
                # Initial guess. Assume saturation. 
                x0=ta[_t,_r,_c]-0. 
                f0=_IMP(x0,ta[_t,_r,_c],td[_t,_r,_c],p[_t,_r,_c])
                if np.abs(f0)<=eps: tw[_t,_r,_c]=x0; continue # Got it first go!
        
                # Second guess, assume Tw=Ta-1    
                xi=x0-1.;
                fi=_IMP(xi,ta[_t,_r,_c],td[_t,_r,_c],p[_t,_r,_c])
                if np.abs(fi)<=eps: tw[_t,_r,_c]=xi; continue # Got it 2nd go!
        
            	# Compute first gradient
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx 
        
               # Iterate while error is too big, and while iteration is in <itermax
                while np.abs(fi)>eps and ni<itermax:
        
                    xi=x0-f0/dfdx # new guess at Tw
                    fi=_IMP\
                        (xi,ta[_t,_r,_c],td[_t,_r,_c],p[_t,_r,_c]) # error from this guess
                    if np.abs(fi)<=eps: tw[_t,_r,_c]=xi; break # Exit if small error
                    dx=(xi-x0)
   
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter
        
                # If it didn't converge, set to nan    
                if ni == itermax: 
                    tw[_t,_r,_c]=np.nan
            
                # If it did converge, but reached a value >ta, set to ta 
                # [can happen when close to saturation and precision accepts
                # solution >ta] 
                elif xi >ta[_t,_r,_c]:    
                    tw[_t,_r,_c]=ta[_t,_r,_c]*1. # V. close to saturation, so set to ta
                
    return tw

@nb.njit(fastmath={"nnan":False},parallel=True)
def _MDI(nt,nr,nc,tw,ta):
	"""
	Tw and Ta should be input in K
	MDI is output in C
	"""
	mdi=np.zeros((nt,nr,nc))*np.nan
	for _t in range(nt):
		for _r in range(nr):
			for _c in range(nc):
				mdi[_t,_r,_c]=(tw[_t,_r,_c]-273.15)*0.75+(ta[_t,_r,_c]-273.15)*0.3
	return mdi  

@nb.njit(fastmath={"nnan":False})
def _IMP_TwTa(tw,t,p,rh):
    
    # _satQ(t,p)
    qtw=_satQ(tw,p)
    qt=_satQ(t,p)
    qt=qt*rh/100.
    Lv,Cp=_LvCp(t,qt)
    diff=(Cp*t+qt*Lv)-(Cp*tw+qtw*Lv)
    
    return diff

@nb.njit(fastmath={"nnan":False},parallel=True)
def _TW_TD_2d(nr,nc,tw,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        cp*Tw + ε*L*esat(Tw)/p = cp*Ta + ε*L*esat(Ta)/p*RH
       
    Using the Newton-Rhapson method
        
    Note that Tw is in K and rh is in %
        
    """
    ta=np.zeros((nr,nc),dtype=np.float32)*np.nan
    itermax=10 # max iterations
    for _r in prange(nr):
        
        for _c in prange(nc):
            
                if np.isnan(tw[_r,_c]) or np.isnan(rh[_r,_c]) or np.isnan(p[_r,_c]): 
                   continue
                
                ni=0 # start iteration value       

                # Initial guess. Assume ta = tw+1 
                x0=tw[_r,_c]+1
                
                # _IMP_TwTa(tw,t,p,rh)
                f0=_IMP_TwTa(tw[_r,_c],x0,p[_r,_c],rh[_r,_c]) # Initial feval
                
                if np.abs(f0)<=eps: ta[_r,_c]=x0; continue # Got it first go!
                # Second guess, assume Ta=Tw+2    
                xi=x0+1.;
                fi=_IMP_TwTa(tw[_r,_c],xi,p[_r,_c],rh[_r,_c])
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx # first gradient
                if np.abs(fi)<=eps: ta[_r,_c]=xi; continue # Got it 2nd go
                while np.abs(fi)>eps and ni<itermax:
                    xi=x0-f0/dfdx # new guess at Ta
                    fi=_IMP_TwTa(tw[_r,_c],xi,p[_r,_c],rh[_r,_c]) # error from this guess                    
                    if np.abs(fi)<=eps: ta[_r,_c]=xi; break # Exit if small error
                    dx=(xi-x0)
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter
                        
                # If it didn't converge, set to nan    
                if ni == itermax: 
                    tw[_r,_c]=np.nan
            
                # If it did converge, but reached a value >ta, set to ta 
                # [can happen when close to saturation and precision accepts
                # solution >ta] 
                elif xi <tw[_r,_c]:    
                    tw[_r,_c]=np.nan # V. close to saturation, so set to tai
    return ta

@nb.njit(fastmath={"nnan":False})
def haversine_fast(lat1,lng1,lat2,lng2,miles=False):
    R = 6378.1370 # equatorial radius of Earth (km)
    """ 
    Calculate the great-circle distance bewteen two points on the Earth surface.

    :input: two scalars (lat1,lng1) and two arrays (lat2,lng2)

    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))

    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.

    """
    # convert all latitudes/longitudes from decimal degrees to radians
    lat1=np.radians(lat1); lat2=np.radians(lat2); lng1=np.radians(lng1); 
    lng2=np.radians(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * R * np.arcsin(np.sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers



# From Lu and Romps
# Version 1.1 released by Yi-Chuan Lu on February 23, 2023.
#    Release 1.1 accommodates old Python 2 installations that
#    interpret some constants as integers instead of reals.
# Version 1.0 released by Yi-Chuan Lu on May 18, 2022.
# 
# When using this code, please cite:
# 
# @article{20heatindex,
#   Title   = {Extending the Heat Index},
#   Author  = {Yi-Chuan Lu and David M. Romps},
#   Journal = {Journal of Applied Meteorology and Climatology},
#   Year    = {2022},
#   Volume  = {61},
#   Number  = {10},
#   Pages   = {1367--1383},
#   Year    = {2022},
# }
#
# This headindex function returns the Heat Index in Kelvin. The inputs are:
# - T, the temperature in Kelvin
# - RH, the relative humidity, which is a value from 0 to 1
# - show_info is an optional logical flag. If true, the function returns the physiological state.

# Thermodynamic parameters
Ttrip = 273.16       # K
ptrip = 611.65       # Pa
E0v   = 2.3740e6     # J/kg
E0s   = 0.3337e6     # J/kg
rgasa = 287.04       # J/kg/K 
rgasv = 461.         # J/kg/K 
cva   = 719.         # J/kg/K
cvv   = 1418.        # J/kg/K 
cvl   = 4119.        # J/kg/K
cvs   = 1861.        # J/kg/K
cpa   = cva + rgasa
cpv   = cvv + rgasv

# The saturation vapor pressure
def pvstar(T):
    if T == 0.0:
        return 0.0
    elif T<Ttrip:
        return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * math.exp( (E0v + E0s -(cvv-cvs)*Ttrip)/rgasv * (1./Ttrip - 1./T) )
    else:
        return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * math.exp( (E0v       -(cvv-cvl)*Ttrip)/rgasv * (1./Ttrip - 1./T) )

# The latent heat of vaporization of water
def Le(T):
    return (E0v + (cvv-cvl)*(T-Ttrip) + rgasv*T)

# Thermoregulatory parameters
sigma       = 5.67e-8                     # W/m^2/K^4 , Stefan-Boltzmann constant
epsilon     = 0.97                        #           , emissivity of surface, steadman1979
M           = 83.6                        # kg        , mass of average US adults, fryar2018
H           = 1.69                        # m         , height of average US adults, fryar2018
A           = 0.202*(M**0.425)*(H**0.725) # m^2       , DuBois formula, parson2014
cpc         = 3492.                       # J/kg/K    , specific heat capacity of core, gagge1972
C           = M*cpc/A                     #           , heat capacity of core
r           = 124.                        # Pa/K      , Zf/Rf, steadman1979
Q           = 180.                        # W/m^2     , metabolic rate per skin area, steadman1979
phi_salt    = 0.9                         #           , vapor saturation pressure level of saline solution, steadman1979
Tc          = 310.                        # K         , core temperature, steadman1979
Pc          = phi_salt * pvstar(Tc)       #           , core vapor pressure
L           = Le(310.)                    #           , latent heat of vaporization at 310 K
p           = 1.013e5                     # Pa        , atmospheric pressure
eta         = 1.43e-6                     # kg/J      , "inhaled mass" / "metabolic rate", steadman1979
Pa0         = 1.6e3                       # Pa        , reference air vapor pressure in regions III, IV, V, VI, steadman1979
hc_glob     = 3.0                         # W/k/m^2   , heat transfer coefficient 

# Thermoregulatory functions
def Qv(Ta,Pa,Q): # respiratory heat loss, W/m^2
    return  eta * Q *(cpa*(Tc-Ta)+L*rgasa/(p*rgasv) * ( Pc-Pa ) )
def Zs(Rs): # mass transfer resistance through skin, Pa m^2/W
    return (52.1 if Rs == 0.0387 else 6.0e8 * Rs**5)
def Ra(Ts,Ta,hc): # heat transfer resistance through air, exposed part of skin, K m^2/W
    hc      = hc/12.3*17.4
    phi_rad = 0.85
    hr      = epsilon * phi_rad * sigma* (Ts**2 + Ta**2)*(Ts + Ta)
    return 1./(hc+hr)
def Ra_bar(Tf,Ta,hc): # heat transfer resistance through air, clothed part of skin, K m^2/W
    hc      = hc/12.3*11.6
    phi_rad = 0.79
    hr      = epsilon * phi_rad * sigma* (Tf**2 + Ta**2)*(Tf + Ta)
    return 1./(hc+hr)
def Ra_un(Ts,Ta,hc): # heat transfer resistance through air, when being naked, K m^2/W
    phi_rad = 0.80
    hr      = epsilon * phi_rad * sigma* (Ts**2 + Ta**2)*(Ts + Ta)
    return 1./(hc+hr)

hc_exp=17.4*hc_glob/12.3
hc_cloth=11.6*hc_glob/12.3
def Za(hc):# Pa m^2/W, mass transfer resistance through air, exposed part of skin
    return 60.6/(17.4*hc/12.3)

def Za_bar(hc):# Pa m^2/W, mass transfer resistance through air, clothed part of skin
    return 60.6/(11.6*hc/12.3)

def Za_un(hc):# Pa m^2/W, mass transfer resistance through air, when being naked
    return 60.6/hc    

# tolerance and maximum iteration for the root solver 
tol     = 1e-8
tolT    = 1e-8
maxIter = 100

# Given air temperature and relative humidity, returns the equivalent variables 
def find_eqvar(Ta,Pa,Q,hc):
    Rs    = 0.0387        # m^2K/W  , heat transfer resistance through skin
    phi   = 0.84          #         , covering fraction
    dTcdt = 0.            # K/s     , rate of change in Tc
    m     = (Pc-Pa)/(Zs(Rs)+Za(hc))
    m_bar = (Pc-Pa)/(Zs(Rs)+Za_bar(hc))
    Ts = solve(lambda Ts: (Ts-Ta)/Ra(Ts,Ta,hc)     + (Pc-Pa)/(Zs(Rs)+Za(hc))     - (Tc-Ts)/Rs, max(0.,min(Tc,Ta)-Rs*abs(m)),     max(Tc,Ta)+Rs*abs(m),    tol,maxIter)
    Tf = solve(lambda Tf: (Tf-Ta)/Ra_bar(Tf,Ta,hc) + (Pc-Pa)/(Zs(Rs)+Za_bar(hc)) - (Tc-Tf)/Rs, max(0.,min(Tc,Ta)-Rs*abs(m_bar)), max(Tc,Ta)+Rs*abs(m_bar),tol,maxIter)
    flux1 = Q-Qv(Ta,Pa,Q)-(1.-phi)*(Tc-Ts)/Rs                   # C*dTc/dt when Rf=Zf=\inf
    flux2 = Q-Qv(Ta,Pa,Q)-(1.-phi)*(Tc-Ts)/Rs - phi*(Tc-Tf)/Rs  # C*dTc/dt when Rf=Zf=0
    if (flux1 <= 0.) : # region I
        eqvar_name = "phi"
        phi = 1.-(Q-Qv(Ta,Pa,Q))*Rs/(Tc-Ts)
        Rf  = float('inf')
    elif (flux2 <=0.) : # region II&III
        eqvar_name = "Rf"
        Ts_bar = Tc - (Q-Qv(Ta,Pa,Q))*Rs/phi + (1./phi -1.)*(Tc-Ts)
        Tf = solve(lambda Tf: (Tf-Ta)/Ra_bar(Tf,Ta,hc) + (Pc-Pa)*(Tf-Ta)/((Zs(Rs)+Za_bar(hc))*(Tf-Ta)+r*Ra_bar(Tf,Ta,hc)*(Ts_bar-Tf)) - (Tc-Ts_bar)/Rs, Ta,Ts_bar,tol,maxIter)
        Rf = Ra_bar(Tf,Ta,hc)*(Ts_bar-Tf)/(Tf-Ta)
    else: # region IV,V,VI
        Rf = 0.
        flux3 =  Q-Qv(Ta,Pa,Q)-(Tc-Ta)/Ra_un(Tc,Ta,hc)-(phi_salt*pvstar(Tc)-Pa)/Za_un(hc)
        if (flux3 < 0.) : # region IV,V
            Ts = solve(lambda Ts: (Ts-Ta)/Ra_un(Ts,Ta,hc)+(Pc-Pa)/(Zs((Tc-Ts)/(Q-Qv(Ta,Pa,Q)))+Za_un(hc))-(Q-Qv(Ta,Pa,Q)),0.,Tc,tol,maxIter)
            Rs = (Tc-Ts)/(Q-Qv(Ta,Pa,Q))
            eqvar_name = "Rs"
            Ps = Pc - (Pc-Pa)* Zs(Rs)/( Zs(Rs)+Za_un(hc))
            if (Ps > phi_salt * pvstar(Ts)):  # region V
                Ts = solve( lambda Ts : (Ts-Ta)/Ra_un(Ts,Ta,hc) + (phi_salt*pvstar(Ts)-Pa)/Za_un(hc) -(Q-Qv(Ta,Pa,Q)), 0.,Tc,tol,maxIter)
                Rs = (Tc-Ts)/(Q-Qv(Ta,Pa,Q))
                eqvar_name = "Rs*"
        else: # region VI
            Rs = 0.
            eqvar_name = "dTcdt"
            dTcdt = (1./C)* flux3
    return [eqvar_name,phi,Rf,Rs,dTcdt]

# given the equivalent variable, find the Heat Index
def find_T(eqvar_name,eqvar):
    if (eqvar_name == "phi"):
        T = solve(lambda T: find_eqvar(T,pvstar(T),180,12.3)[1]-eqvar,0.,240.,tolT,maxIter)
        region = 'I'
    elif (eqvar_name == "Rf"):
        T = solve(lambda T: find_eqvar(T,min(Pa0,pvstar(T)),180,12.3)[2]-eqvar,230.,300.,tolT,maxIter)
        region = ('II' if Pa0>pvstar(T) else 'III')
    elif (eqvar_name == "Rs" or eqvar_name == "Rs*"):
        T = solve(lambda T: find_eqvar(T,Pa0,180,12.3)[3]-eqvar,295.,350.,tolT,maxIter)
        region = ('IV' if eqvar_name == "Rs" else 'V')
    else:
        T = solve(lambda T: find_eqvar(T,Pa0,180,12.3)[4]-eqvar,340.,1000.,tolT,maxIter)
        region = 'VI'
    return T, region

# combining the two functions find_eqvar and find_T
def heatindex(Ta,RH,Q,hc,show_info=False):
    
    dic = {"phi":1,"Rf":2,"Rs":3,"Rs*":3,"dTcdt":4}
    Pa = pvstar(Ta)*RH
    eqvars = find_eqvar(Ta,Pa,Q,hc)
    T, region = find_T(eqvars[0],eqvars[dic[eqvars[0]]])
    if (Ta == 0.): T = 0.
    if (show_info==True):
        if region=='I':
            print("Region I, covering (variable phi)")
            print("Clothing fraction is "+ str(round(eqvars[1],3)))
        elif region=='II':
            print("Region II, clothed (variable Rf, pa = pvstar)")
            print("Clothing thickness is "+ str(round((eqvars[2]/16.7)*100.,3))+" cm")
        elif region=='III':
            print("Region III, clothed (variable Rf, pa = pref)")
            print("Clothing thickness is "+ str(round((eqvars[2]/16.7)*100.,3))+" cm")
        elif region=='IV':
            kmin = 5.28               # W/K/m^2     , conductance of tissue
            rho  = 1.0e3              # kg/m^3      , density of blood
            c    = 4184.              # J/kg/K      , specific heat of blood
            print("Region IV, naked (variable Rs, ps < phisalt*pvstar)")
            print("Blood flow is " + str(round(( (1./eqvars[3] - kmin)*A/(rho*c) ) *1000.*60.,3))+" l/min")
        elif region=='V':
            kmin = 5.28               # W/K/m^2     , conductance of tissue
            rho  = 1.0e3              # kg/m^3      , density of blood
            c    = 4184.              # J/kg/K      , specific heat of blood
            print("Region V, naked dripping sweat (variable Rs, ps = phisalt*pvstar)")
            print("Blood flow is " + str(round(( (1./eqvars[3] - kmin)*A/(rho*c) ) *1000.*60.,3))+" l/min")
        else:
            print("Region VI, warming up (dTc/dt > 0)")
            print("dTc/dt = "+ str(round(eqvars[4]*3600.,6))+ " K/hour")
    return T

def solve(f,x1,x2,tol,maxIter):
    a  = x1
    b  = x2
    fa = f(a)
    fb = f(b)
    if fa*fb>0.:
        raise SystemExit('wrong initial interval in the root solver')
        return None
    else:
        for i in range(maxIter):
            c  = (a+b)/2.
            fc = f(c)
            if fb*fc > 0. :
                b  = c
                fb = fc
            else:
                a  = c
                fa = fc   
            if abs(a-b) < tol:
                return c
            if i == maxIter-1:
                raise SystemExit('reaching maximum iteration in the root solver')
                return None
            


# SHORT HEADER
#
# [Twb,Teq,epott]=WetBulb(TemperatureC,Pressure,Humidity,[HumidityMode])
#
# Calculate wet-bulb temperature, equivalent temperature, and equivalent
# potential temperature from air temperature, atmospheric pressure, and specific humidity.
#
# Required input units: air temperature in C, pressure in Pa, specific humidity in kg/kg
# Output: wet-bulb temperature in C, equivalent temperature and equivalent potential temperature in K
#
# Example usage for a single value:
# From a Jupyter notebook
# Twb=WetBulb(25.,100000.,0.015,0)[0]     #should return 21.73 C
#
# From the command line
# python speedywetbulb.py 25. 100000. 0.015 > out.txt   #ditto

# Runtime on a MacBook Pro: approximately 0.3 sec for 10^6 calculations



# DETAILED HEADER
#
# Calculates wet-bulb temperature and associated variables using the Davies-Jones 2008 method.
# This entails calculating the lifting condensation temperature (Bolton 1980 eqn 22),
# then the moist potential temperature (Bolton 1980 eqn 24), 
# then the equivalent potential temperature (Bolton 1980 eqn 39),
# and finally, from equivalent potential temp, equivalent temp and theta_w (Davies-Jones 2008 eqn 3.5-3.8), 
# an accurate 'first guess' of wet-bulb temperature (Davies-Jones 2008 eqn 4.8-4.11). 
# The Newton-Raphson method is used for 2 iterations, 
# to obtain the final calculated wet-bulb temperature (Davies-Jones 2008 eqn 2.6).
#
# Reference:  Bolton: The computation of equivalent potential temperature.
# 	      Monthly weather review (1980) vol. 108 (7) pp. 1046-1053
#	      Davies-Jones: An efficient and accurate method for computing the
#	      wet-bulb temperature along pseudoadiabats. Monthly Weather Review
#	      (2008) vol. 136 (7) pp. 2764-2785
# 	      Flatau et al: Polynomial fits to saturation vapor pressure.
#	      Journal of Applied Meteorology (1992) vol. 31 pp. 1507-1513

#
# Ported from HumanIndexMod by Jonathan R Buzan, April 2016
# Ported to Python by Xianxiang Li, February 2019
#
# Further optimizations with numba and bug correction applied by Alex Goodman, April 2023,
# with consultation and inline comments by Colin Raymond

# Additional bugs noticed and corrections proposed by Rob Warren, 
# implemented here by Colin Raymond, August 2023

# Set constants

SHR_CONST_TKFRZ = np.float64(273.15)
lambd_a = np.float64(3.504)    	# Inverse of Heat Capacity
alpha = np.float64(17.67) 	    # Constant to calculate vapor pressure
beta = np.float64(243.5)		# Constant to calculate vapor pressure
epsilon = np.float64(0.6220)	# Conversion between pressure/mixing ratio
es_C = np.float64(611.2)		# Vapor Pressure at Freezing STD (Pa)
y0 = np.float64(3036)		    # constant
y1 = np.float64(1.78)		    # constant
y2 = np.float64(0.448)		    # constant
Cf = SHR_CONST_TKFRZ	# Freezing Temp (K)
p0 = np.float64(100000)	    # Reference Pressure (Pa)
constA = np.float64(2675) 	 # Constant used for extreme cold temperatures (K)
vkp = np.float64(0.2854)	 # Heat Capacity




#Define QSat_2 function

@nb.njit(fastmath=True)
def QSat_2(T_k, p_t, p0ndplam):
    # Constants used to calculate es(T)
    # Clausius-Clapeyron
    tcfbdiff = T_k - Cf + beta
    es = es_C * np.exp(alpha*(T_k - Cf)/(tcfbdiff))
    dlnes_dT = alpha * beta/((tcfbdiff)*(tcfbdiff))
    pminuse = p_t - es
    de_dT = es * dlnes_dT

    # Constants used to calculate rs(T)
    rs = epsilon * es/(p0ndplam - es + np.spacing(1)) #eps

    # avoid bad numbers
    if rs > 1 or rs < 0:
        rs = np.nan
        
    return es,rs,dlnes_dT 



#Define main wet-bulb-temperature function

@nb.njit(fastmath=True)
def WetBulb(TemperatureC,Pressure,Humidity,HumidityMode=1):
    ###
    #Unless necessary, default to using specific humidity as input (simpler and tends to reduce error margins)#
    ###

    """
    INPUTS:
      TemperatureC	   2-m air temperature (degrees Celsius)
      Pressure	       Atmospheric Pressure (Pa)
      Humidity         Humidity -- meaning depends on HumidityMode
      HumidityMode
        0 (Default): Humidity is specific humidity (kg/kg)
        1: Humidity is relative humidity (#, max = 100)
      TemperatureC, Pressure, and Humidity should either be scalars or arrays of identical dimension.
    OUTPUTS:
      Twb	    wet bulb temperature (C)
      Teq	    Equivalent Temperature (K)
      epott 	Equivalent Potential Temperature (K)
    """
    TemperatureK = TemperatureC + SHR_CONST_TKFRZ
    pnd = (Pressure/p0)**(vkp)
    p0ndplam = p0*pnd**lambd_a

    C = SHR_CONST_TKFRZ;		# Freezing Temperature
    T1 = TemperatureK;		# Use holder for T

    es, rs, _ = QSat_2(TemperatureK, Pressure, p0ndplam) # first two returned values
    

    if HumidityMode==0:
        qin = Humidity                   # specific humidity
        mixr = (qin / (1-qin))           # corrected by Rob Warren
        vape = (Pressure * mixr) / (epsilon + mixr) #corrected by Rob Warren
        relhum = 100.0 * vape/es         # corrected by Rob Warren
    elif HumidityMode==1:
        relhum = Humidity                # relative humidity (%)
        vape = es * relhum * 0.01        # vapor pressure (Pa)
        mixr = epsilon * vape / (Pressure-vape)  #corrected by Rob Warren

    mixr = mixr * 1000
    
    # Calculate Equivalent Pot. Temp (Pressure, T, mixing ratio (g/kg), pott, epott)
    # Calculate Parameters for Wet Bulb Temp (epott, Pressure)
    D = 1.0/(0.1859*Pressure/p0 + 0.6512)
    k1 = -38.5*pnd*pnd + 137.81*pnd - 53.737
    k2 = -4.392*pnd*pnd + 56.831*pnd - 0.384

    # Calculate lifting condensation level
    tl = (1.0/((1.0/((T1 - 55))) - (np.log(relhum/100.0)/2840.0))) + 55.0

    # Theta_DL: Bolton 1980 Eqn 24.
    theta_dl = T1*((p0/(Pressure-vape))**vkp) * ((T1/tl)**(mixr*0.00028))
    # EPT: Bolton 1980 Eqn 39.
    epott = theta_dl * np.exp(((3.036/tl)-0.00178)*mixr*(1 + 0.000448*mixr))
    Teq = epott*pnd	# Equivalent Temperature at pressure
    X = (C/Teq)**3.504
    
    # Calculates the regime requirements of wet bulb equations.
    invalid = Teq > 600 or Teq < 200
    hot = Teq > 355.15
    cold = X<1   #corrected by Rob Warren
        
    if invalid:
        return np.nan, np.nan, epott

    # Calculate Wet Bulb Temperature, initial guess
    # Extremely cold regimes: if X.gt.D, then need to calculate dlnesTeqdTeq

    es_teq, rs_teq, dlnes_dTeq = QSat_2(Teq, Pressure, p0ndplam)
    if X<=D:
        wb_temp = C + (k1 - 1.21 * cold - 1.45 * hot - (k2 - 1.21 * cold) * X + (0.58 / X) * hot)
    else:
        wb_temp = Teq - ((constA*rs_teq)/(1 + (constA*rs_teq*dlnes_dTeq)))

    # Newton-Raphson Method
    maxiter = 2
    iter = 0
    delta = 1e6

    while delta>0.01 and iter<maxiter:
        foftk_wb_temp, fdwb_temp = DJ(wb_temp, Pressure, p0ndplam)
        delta = (foftk_wb_temp - X)/fdwb_temp
        delta = np.minimum(10,delta)
        delta = np.maximum(-10,delta) #max(-10,delta)
        wb_temp = wb_temp - delta
        Twb = wb_temp
        iter = iter+1

    Tw_final=np.round(Twb-C,2)
    
    return Tw_final,Teq,epott



# Define parallelization functions for wet-bulb (optional)

@nb.njit(fastmath=True)
def WetBulb_all(tempC, Pres, relHum, Hum_mode=1):
    Twb = np.empty_like(tempC)
    Teq = np.empty_like(tempC)
    epott = np.empty_like(tempC)
    for i in nb.prange(Twb.size):
        Twb[i], Teq[i], epott[i] = WetBulb(tempC[i], Pres[i], relHum[i], Hum_mode)
    return Twb

@nb.njit(fastmath=True, parallel=True)
def WetBulb_par(tempC, Pres, relHum, Hum_mode):
    Twb = np.empty_like(tempC)
    Teq = np.empty_like(tempC)
    epott = np.empty_like(tempC)
    for i in nb.prange(Twb.size):
        Twb[i], Teq[i], epott[i] = WetBulb(tempC[i], Pres[i], relHum[i], Hum_mode)



# Define helper functions for usage in the Davies-Jones wet-bulb algorithm

@nb.njit(fastmath=True)
def DJ(T_k, p_t, p0ndplam):
    # Constants used to calculate es(T)
    # Clausius-Clapeyron
    tcfbdiff = T_k - Cf + beta
    es = es_C * np.exp(alpha*(T_k - Cf)/(tcfbdiff))
    dlnes_dT = alpha * beta/((tcfbdiff)*(tcfbdiff))
    pminuse = p_t - es
    de_dT = es * dlnes_dT

    # Constants used to calculate rs(T)
    rs = epsilon * es/(p0ndplam - es + np.spacing(1)) #eps)
    prersdt = epsilon * p_t/((pminuse)*(pminuse))
    rsdT = prersdt * de_dT

    # Constants used to calculate g(T)
    rsy2rs2 = rs + y2*rs*rs
    oty2rs = 1 + 2.0*y2*rs
    y0tky1 = y0/T_k - y1
    goftk = y0tky1 * (rs + y2 * rs * rs)
    gdT = - y0 * (rsy2rs2)/(T_k*T_k) + (y0tky1)*(oty2rs)*rsdT

    # Calculations for calculating f(T,ndimpress)
    foftk = ((Cf/T_k)**lambd_a)*(1 - es/p0ndplam)**(vkp*lambd_a)* \
        np.exp(-lambd_a*goftk)
    fdT = -lambd_a*(1.0/T_k + vkp*de_dT/pminuse + gdT) * foftk  #derivative corrected by Qinqin Kong

    return foftk,fdT


