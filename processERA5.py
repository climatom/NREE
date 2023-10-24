#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:32:42 2023

@author: k2147389
"""

import pickle as pk, numpy as np, xarray as xa, matplotlib.pyplot as plt,\
    utils as ut
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numba as nb, pandas as pd
from numba import prange
import cartopy.crs as ccrs
import matplotlib.colors as colors
from scipy.stats import rankdata
import matplotlib.ticker as tkr
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Fast, explicit function to determine indices in lookup array
#@nb.jit("float64[:](float64[:],float64[:],float64[:,:],float64[:,:],float64[:,:])")
def NN(lat,lon,latref,lonref,hiref):
    nt=len(lat)
    out=np.zeros(nt)*np.nan
    lat_st=np.max(latref)
    lon_st=np.min(lonref)
    res_lat=latref[0,0]-latref[1,0]; 
    res_lon=lonref[0,1]-lonref[0,0]; 
    for ti in range(nt):
        i=np.floor((lat_st-lat[ti])/res_lat).astype(int) # row
        j=np.floor((lon[ti]-lon_st)/res_lon).astype(int) # Row
               
        # Row = i, col = j
        out[ti]=hiref[i,j]
                
    return out    

def WETBULB_T(ta,rh,tw):
    ta=float(ta)
    rh=float(rh)
    tw=float(tw)

    dt=np.abs(tw-ut.WetBulb(ta,101300.,rh)[0])

    return dt

def WETBULB_RH(rh,ta,tw):
    ta=float(ta)
    rh=float(rh)
    tw=float(tw)

    dt=np.abs(tw-ut.WetBulb(ta,101300.,rh)[0])

    return dt

def HI_RH(rh,ta,hi):
    ta=float(ta)
    rh=float(rh)
    hi=float(hi)
    ut.heatindex(ta,rh,82.9,3.0)  
    dt=np.abs(hi-ut.heatindex(ta,rh,82.9,3.0))
    return dt

  
def CRIT_RH(ta,tw):
    # (swrate,ta,rh,met,va)
    #WetBulb(TemperatureC,Pressure,Humidity,HumidityMode=0)
    out=minimize(fun=WETBULB_RH,
                          x0=20.,
                          args=(ta,tw),
                          method="Nelder-Mead",
                          options={'tol':1e-6})    
    return out


def CRIT_RH_HI(ta,hi):
    # (swrate,ta,rh,met,va)
    #WetBulb(TemperatureC,Pressure,Humidity,HumidityMode=0)
    out=minimize(fun=HI_RH,
                          x0=0.5,
                          args=(ta,hi),
                          method="Nelder-Mead",
                          options={'tol':1e-6})    
    return out

# Set parameters 
yst=1993
ystp=1995
d="../ERA5/"
plot_d="../Plots/"
template=pk.load(open(d+"himax_2000.p",'rb'))
nr,nc=template.shape
nt=ystp-yst+1
res_rh=0.005
res_ta=0.05
hi_lower=37.2 # Set anything < to NaN for plotting
hi_thresh=71.5+273.15
tw_thresh=35.0+273.15
test_plot=False



# Ref download -- to track lat/lon
refclim=xa.open_dataset("../ERA5/download.nc")
reflat=refclim.latitude.data[:]
reflon=refclim.longitude.data[:]
lon2,lat2=np.meshgrid(reflon,reflat)

# Read in
landsea=xa.open_dataset(d+'landsea.nc')
land=np.squeeze(landsea.lsm.data)

# Import city data
city=pd.read_csv("../city_ref.csv")

# Preallocate 
vs=["ta","tw","hi"]
arc={}
hi_ta=np.zeros((nt,nr,nc))
hi_rh=np.zeros((nt,nr,nc))

maxima={}
vanos={}


# Read in
for v in vs:
    arc[v]=np.zeros((nt,nr,nc))
    for y in range(yst,ystp+1):    
        arc[v][y-yst,:,:]=pk.load(open(d+"%s_max_%.0f.p"%(v,y),'rb'))
        if v != "tw": 
            off=-273.15
        else: off = 0
        arc[v][y-yst,:,:]+=off
        fldmax=np.nanmax(arc[v][y-yst,:,:])
        print("Max %s in year: %.0f = %.2fC"%(v,y,fldmax))
        if v=="hi":
            hi_ta[y-yst,:,:]=pk.load(open(d+"%s_tamax_%.0f.p"%(v,y),'rb'))-273.15
            hi_rh[y-yst,:,:]=pk.load(open(d+"%s_rhmax_%.0f.p"%(v,y),'rb'))*100.
        if test_plot:
            levs=np.linspace(np.nanpercentile(arc[v],1),np.nanmax(arc[v]),50)
            fig, ax = plt.subplots(1,1)
            ax.contourf(lon2, lat2, arc[v][y-yst,:,:],cmap="turbo",levels=levs)
            fig.savefig(plot_d+"%s_%.0f.png"%(v,y))
    _=np.nanmax(arc[v],axis=0)
    maxima[v]=_
    maxima[v+"_norm"]=rankdata(_).reshape(_.shape)/_.size
    
hi_idx=np.nanargmax(arc["hi"],axis=0)
hi_ta=np.squeeze(np.take_along_axis(hi_ta,hi_idx[None,:,:],axis=0)).flatten()
hi_rh=np.squeeze(np.take_along_axis(hi_rh,hi_idx[None,:,:],axis=0)).flatten()
    
    
    
vanos_old={}
vanos_young={}
vanos_death_hours_old=np.zeros((nr,nc))
vanos_death_hours_young=np.zeros((nr,nc))

for i in range(3,6):
    vanos_old[i]=np.zeros((nt,nr,nc))
    vanos_young[i]=np.zeros((nt,nr,nc))
    
    for y in range(yst,ystp+1): 
        vanos_old[i][y-yst,:,:]=pk.load(open(d+"vanos_old_%.0f_%.0f.p"%(i,y),'rb'))
        vanos_young[i][y-yst,:,:]=pk.load(open(d+"vanos_young_%.0f_%.0f.p"%(i,y),'rb'))
        
    vanos_death_hours_old+=np.sum(vanos_old[i],axis=0)/(ystp-yst+1)
    vanos_death_hours_young+=np.sum(vanos_young[i],axis=0)/(ystp-yst+1)

city["old_dead_hy"]= NN(city["lat"].values[:],city["lon"].values[:],\
           lat2,lon2,vanos_death_hours_old)
city["young_dead_hy"]= NN(city["lat"].values[:],city["lon"].values[:],\
            lat2,lon2,vanos_death_hours_young)   
city["death_risk_old"]=city["old_dead_hy"]*city["pop"]
city["death_risk_young"]=city["young_dead_hy"]*city["pop"]
city.sort_values(by="death_risk_old",ascending=False,inplace=True)
    
vanos_death_hours_old[vanos_death_hours_old==0]=np.nan
vanos_death_hours_young[vanos_death_hours_young==0]=np.nan

# Plot world map(s)
## First, abs quantitites
fig, axs = plt.subplots(nrows=2,ncols=2,
                        subplot_kw={'projection': ccrs.EckertIII()},
                        figsize=(11,6.5))
# Drybulb
axs=axs.flatten()
ctd=axs[0].contourf(lon2, lat2, maxima["ta"], 
                    transform=ccrs.PlateCarree(),
                    transform_first=True,
                    cmap="turbo",
                    levels=50,)
axs[0].coastlines('110m', alpha=0.5)
cb1=plt.colorbar(ctd,ax=axs[0],orientation='horizontal',shrink=0.75,pad=0.05,
                 format=tkr.FormatStrFormatter('%.1f'))
cb1.set_label("Ta ($^{\circ}$C)", labelpad=10, y=1.05, rotation=0)
cb1.set_ticks([7,14,21,28,35,42,49])


# Wetbulb
ctw=axs[1].contourf(lon2, lat2, maxima["tw"], 
                    transform=ccrs.PlateCarree(),
                    transform_first=True,
                    cmap="turbo",
                    levels=50,)
axs[1].coastlines('110m', alpha=0.5)
cb2=plt.colorbar(ctw,ax=axs[1],orientation='horizontal',shrink=0.75,pad=0.05,
                 format=tkr.FormatStrFormatter('%.1f'))
cb2.set_label("Tw ($^{\circ}$C)", labelpad=10, y=1.05, rotation=0)
cb2.set_ticks([7,12,17,22,27,32])


## Normalised diff
z=maxima["tw_norm"]-maxima["ta_norm"]
levs=np.linspace(-0.7,0.7,50)
c=axs[2].contourf(lon2, lat2, z, transform=ccrs.PlateCarree(),transform_first=True,
                norm=colors.CenteredNorm(),cmap="turbo",levels=levs)
axs[2].coastlines('110m', alpha=0.5)
cb3=plt.colorbar(c,ax=axs[2],orientation='horizontal',shrink=0.75,pad=0.05,\
                 format=tkr.FormatStrFormatter('%.1f'))
cb3.set_label("Tw-Ta ($^{\circ}$C)", labelpad=10, y=1.05, rotation=0)
cb3.set_ticks([-0.5,-0.3,-0.1,0.1,0.3,0.5])

# Latitudinal aspects
axs[-1].remove()
ax=fig.add_subplot(2,2,4)
ta_lat=np.nanpercentile(maxima["ta_norm"],99.9,axis=1)
tw_lat=np.nanpercentile(maxima["tw_norm"],99.9,axis=1)
ta=ax.plot(reflat,ta_lat,color='red',label="Ta")
td=ax.plot(reflat,tw_lat,color='blue',label="Tw")
ax.grid()
ax.set_xlabel("Latitude [$^{\circ}$ north]")
ax.set_ylabel("Normalised extreme ")
ax.legend()

fig.savefig(plot_d+"ERA5.png",dpi=300)


# Critical RH values
vl=pd.read_csv("../Vecellio_lims.csv")
rh_vl=np.zeros(len(vl))*np.nan
for ii in range(len(vl)):
    rh_vl[ii]=CRIT_RH(vl["ta"][ii],vl["tw"][ii]).x[0]

# Critical values for Powis
ta_pow=np.linspace(25,60,100)
rh_pow=0.0947376*ta_pow**2-12.11*ta_pow+381.787

rh_lu=np.zeros(len(ta_pow))
for ii in range(len(ta_pow)):
   rh_lu[ii]=CRIT_RH_HI(ta_pow[ii]+273.15,71.5+273.15).x[0]*100.
   
rh_35 =np.zeros(len(ta_pow)) 
for ii in range(len(ta_pow)):
   rh_35[ii]=CRIT_RH(ta_pow[ii],35.).x[0]

fig,ax=plt.subplots(1,1)
ax.scatter(hi_ta,hi_rh,s=0.001,c='k')
ax.plot(vl["ta"],rh_vl)
ax.plot(ta_pow,rh_pow)
ax.plot(ta_pow,rh_lu)
ax.plot(ta_pow,rh_35)
ax.set_xlim(25,55)
ax.set_ylim(0,100)
ax.grid()


# Now go back up and: (1) extract the T and RH during the max HI. Plot this
# Also include a moving average: rh as f(T)


# Repeat for the Heat Index and Vanos
# HI
# fig, axs = plt.subplots(nrows=3,ncols=1,
#                         subplot_kw={'projection': ccrs.EckertIII()},
#                         figsize=(6,12))
# axs=axs.flatten()
# z=maxima["hi"]-71.5
# z[z<0]=np.nan
# c=axs[0].contourf(lon2, lat2, z, transform=ccrs.PlateCarree(),transform_first=True,
#                 norm=colors.CenteredNorm(),cmap="turbo",levels=50)
# cb4=plt.colorbar(c,ax=axs[0],orientation='horizontal',shrink=0.75,pad=0.05,\
#                  format=tkr.FormatStrFormatter('%.1f'))
# axs[0].coastlines('110m', alpha=0.5)
 
# # Vanos young
# c=axs[1].contourf(lon2, lat2, vanos_death_hours_young, transform=ccrs.PlateCarree(),transform_first=True,
#                 norm=colors.CenteredNorm(),cmap="turbo",levels=50)

# cb5=plt.colorbar(c,ax=axs[1],orientation='horizontal',shrink=0.75,pad=0.05,\
#                  format=tkr.FormatStrFormatter('%.1f'))
# axs[1].coastlines('110m', alpha=0.5)

# # Vanos old
# c=axs[2].contourf(lon2, lat2, vanos_death_hours_old, transform=ccrs.PlateCarree(),transform_first=True,
#                 norm=colors.CenteredNorm(),cmap="turbo",levels=50)
# cb6=plt.colorbar(c,ax=axs[2],orientation='horizontal',shrink=0.75,pad=0.05,\
#                  format=tkr.FormatStrFormatter('%.1f'))
# axs[2].coastlines('110m', alpha=0.5)




