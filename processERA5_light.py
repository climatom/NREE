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
from scipy.stats import binned_statistic
import matplotlib.ticker as tkr
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import xesmf as xe

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


@nb.jit(nopython=True,fastmath=True)
def TaTd2HI(ta,td,hiref,taref,tdref):
    nr,nc=ta.shape
    out=np.zeros((nr,nc))*np.nan
    t_st=np.min(taref)
    td_st=np.min(tdref)
    res_t=taref[0,1]-taref[0,0]
    res_td=tdref[1,0]-tdref[0,0]

    for row in nb.prange(nr):
        for col in nb.prange(nc):
            i=int(np.floor((ta[row,col]-t_st)/res_t)) # Col
            j=int(np.floor((td[row,col]-td_st)/res_td)) # Row
                
            # Might be that some cold obs fall off the lower end of the 
            # lookup table. Protect against that here. 
            if i < 0: i=0
            if j < 0: j=0
                
            # Note that ref's columns are ta; rows are td
            out[row,col]=hiref[j,i]
                
    return out  

# Set parameters 
yst=1940
ystp=2022
yrs=np.arange(yst,ystp+1)
p1=1950
p2=2000
window=20
lookup_dir="/Users/k2147389/Desktop/Papers/homeostasis/output/"
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
res_rh=0.005
res_ta=0.05

# ../ta_lookup_ta%.3f_rh%.3f.p"%(res_ta,res_rh)
hiref=pk.load( open( lookup_dir+"HI_lookup_ta%.3f_rh%.3f.p"%(res_ta,res_rh), "rb" ) )
taref=pk.load( open( lookup_dir+"ta_lookup_ta%.3f_rh%.3f.p"%(res_ta,res_rh), "rb" ) )
tdref=pk.load( open( lookup_dir+"td_lookup_ta%.3f_rh%.3f.p"%(res_ta,res_rh), "rb" ) )


# Ref download -- to track lat/lon
refclim=xa.open_dataset("../ERA5/download.nc")
reflat=refclim.latitude.data[:]
reflon=refclim.longitude.data[:]
lon2,lat2=np.meshgrid(reflon,reflat)


# Preallocate 
vs=["ta","tw","hi"]
vs_x=["rh","ta","p"]
arc={}

maxima={}
vanos={}

# Read in
maxs=np.zeros((ystp-yst+1,len(vs)))
vcount=0
for v in vs:
    arc[v]=np.zeros((nt,nr,nc))
    for vx in vs_x: 
        if v == "ta" and vx =="ta": continue
        arc[v+"_"+vx]=np.zeros((nt,nr,nc))
    
    ycount=0
    
    for y in range(yst,ystp+1):    
        arc[v][y-yst,:,:]=pk.load(open(d+"%s_max_%.0f.p"%(v,y),'rb'))
        if v != "tw": 
            off=-273.15
        else: off = 0
        arc[v][y-yst,:,:]+=off
        
        # Compute the max
        fldmax=np.nanmax(arc[v][y-yst,:,:])
        print("Max %s in year: %.0f = %.2fC"%(v,y,fldmax))
        maxs[ycount,vcount]=fldmax
        
        # Load the extras
        for vx in vs_x:        
            if v == "ta" and vx =="ta": continue
            arc[v+"_"+vx][y-yst,:,:]=\
            pk.load(open(d+"%s_%smax_%.0f.p"%(v,vx,y),'rb'))
            if vx == "ta": arc[v+"_"+vx][y-yst,:,:]+=off
        
        if test_plot:
            levs=np.linspace(np.nanpercentile(arc[v],1),np.nanmax(arc[v]),50)
            fig, ax = plt.subplots(1,1)
            ax.contourf(lon2, lat2, arc[v][y-yst,:,:],cmap="turbo",levels=levs)
            fig.savefig(plot_d+"%s_%.0f.png"%(v,y))
        
   
        ycount+=1
        
    
    vcount+=1
    
    
# Write NetCDF files 
for v in vs:
    if v !="ta":
        
        data_x=xa.Dataset({
            
            v:(("year","lat","lon"),arc[v],{"units":"degrees_celsius"}),
            
            v+"_ta":(("year","lat","lon"),arc[v+"_ta"],{"units":"degrees_celsius"}),
     
            v+"_rh":(("year","lat","lon"),arc[v+"_rh"],{"units":"proportion"}),
 
            v+"_p":(("year","lat","lon"),arc[v+"_p"],{"units":"Pascal"}),
       
            },
            coords={"year":yrs,"lat":reflat,"lon":reflon}
            )
    else:
        
        data_x=xa.Dataset({
            
            v:(("year","lat","lon"),arc[v]),
     
            v+"_rh":(("year","lat","lon"),arc[v+"_rh"],{"units":"proportion"}),
 
            v+"_p":(("year","lat","lon"),arc[v+"_p"],{"units":"Pascal"}),
       
            },
            coords={"year":yrs,"lat":reflat,"lon":reflon}
            )
        
    data_x["year"]=data_x.year.assign_attrs(units='years_since 00/00/00')
    data_x["lat"]=data_x.lat.assign_attrs(units='degrees_north')
    data_x["lon"]=data_x.lon.assign_attrs(units='degrees_east')
            
    data_x.to_netcdf(lookup_dir+v+"_max.nc")                          






