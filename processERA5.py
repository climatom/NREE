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
tfile="../hadcrut5.csv"
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

# Read in
landsea=xa.open_dataset(d+'landsea.nc')
land=np.squeeze(landsea.lsm.data)

# Global mean T 
globt=pd.read_csv(tfile,index_col=0)
# Re-normalise
globt_pi=np.mean(globt.iloc[:30,:])
globt=globt["t"].loc[np.logical_and(globt.index>=yst,globt.index<=ystp)]-globt_pi

# Load population
# ssp3, 2020
pop20=xa.open_dataset('../ssp3_2020.nc')
pop20_out = xa.Dataset(
    {
        "latitude": (["latitude"], refclim.latitude.data[:], {"units": "degrees_north"}),
        "longitude": (["longitude"], refclim.longitude.data[:], {"units": "degrees_east"}),
    }
)
regridder20=xe.Regridder(pop20, pop20_out, "bilinear")
pop20=regridder20(pop20)
# ssp3, 2050
pop50=xa.open_dataset('../ssp3_2050.nc')
pop50_out=xa.Dataset(
    {
        "lat": (["lat"], refclim.latitude.data[:], {"units": "degrees_north"}),
        "lon": (["lon"], refclim.longitude.data[:], {"units": "degrees_east"}),
    }
)  
regridder50=xe.Regridder(pop50, pop20_out, "bilinear")    
pop50=regridder20(pop50)

# Import city data
city=pd.read_csv("../city_ref.csv")

# Preallocate 
vs=["ta","tw","hi"]
vs_x=["rh","ta","p"]
arc={}
hi_ta=np.zeros((nt,nr,nc))
hi_rh=np.zeros((nt,nr,nc))

maxima={}
vanos={}


# Read in
maxs=np.zeros((ystp-yst+1,len(vs)))
rhs=np.zeros((ystp-yst+1,1))
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
            
            v:(("year","lat","lon"),arc[v]),
            
            v+"_ta":(("year","lat","lon"),arc[v+"_ta"]),
     
            v+"_rh":(("year","lat","lon"),arc[v+"_rh"]),
 
            v+"_p":(("year","lat","lon"),arc[v+"_p"]),
       
            },
            coords={"year":yrs,"lat":reflat,"lon":reflon}
            )
    else:
        
        data_x=xa.Dataset({
            
            v:(("year","lat","lon"),arc[v]),
     
            v+"_rh":(("year","lat","lon"),arc[v+"_rh"]),
 
            v+"_p":(("year","lat","lon"),arc[v+"_p"]),
       
            },
            coords={"year":yrs,"lat":reflat,"lon":reflon}
            )
            

### NOTE -- need to amend so that we no longer refer to hi_rh/ta... instead, 
### we should use arc[hi_rh] etc.    
assert 1==2
# Here we apply a rolling mean (using convlution to the time dimension of the HI)
# Note that we scale max T corresponding to max HI
kernel = np.ones(window) / window 
hi_ta_s=np.apply_along_axis(np.convolve, axis=0, arr=hi_ta, v=kernel, mode='valid')
# Also smooth the global mean series
globt_s=globt.rolling(window,center=True).mean().dropna()

# Fit regressions between hi_ta_s and glob_t
intercepts,slopes,rs=ut.fast_regress3d(globt_s.values[:],hi_ta_s,nr,nc)
s,i=np.polyfit(globt_s,hi_ta_s[:,0,0],1)
# Now take the max of the last smooth years 


# Here process into two sub periods (early and late). Note that the late 'window'
# contains the t we want to scale
ys=np.arange(yst,ystp+1)
idx_p1=np.logical_and(ys>=p1,ys<=p1+window)
idx_p2=np.logical_and(ys>=p1,ys<=p2+window)
hi_p1=arc["hi"][idx_p1,:,:]
hi_p2=arc["hi"][idx_p2,:,:]

# These cubes will be overwritten with time maxima
hi_ta_p1=hi_ta[idx_p1,:,:]
hi_ta_p2=hi_ta[idx_p2,:,:] # target for scaling
hi_rh_p1=hi_rh[idx_p1,:,:]
hi_rh_p2=hi_rh[idx_p2,:,:] # target for scaling

# Get indices of max hi in each slice 
hi_idx_p1=np.nanargmax(arc["hi"][idx_p1,:,:],axis=0)
hi_idx_p2=np.nanargmax(arc["hi"][idx_p2,:,:],axis=0)

# Redefine now as the max (now 2d)
hi_ta_p1=np.squeeze(np.take_along_axis(hi_ta_p1,hi_idx_p1[None,:,:],axis=0))
hi_rh_p1=np.squeeze(np.take_along_axis(hi_rh_p1,hi_idx_p1[None,:,:],axis=0))
hi_ta_p2=np.squeeze(np.take_along_axis(hi_ta_p2,hi_idx_p2[None,:,:],axis=0)) # ta target for scaling
hi_rh_p2=np.squeeze(np.take_along_axis(hi_rh_p2,hi_idx_p2[None,:,:],axis=0))# rh target for scaling

# Scale the ta using regression
# Also create the plots here. 
scaled={}
ta_pow=np.linspace(25,60,100)
rh_35 =np.zeros(len(ta_pow)) 
rh_lu=np.zeros(len(ta_pow))
ws=[0,1.5,2,4,8]
bins=[150,150,150,200,300]
frac20=np.zeros(len(ws))*np.nan
frac50=np.zeros(len(ws))*np.nan
wcount=0
for ii in range(len(ta_pow)): 
    rh_35[ii]=CRIT_RH(ta_pow[ii],35.).x[0]
    rh_lu[ii]=CRIT_RH_HI(ta_pow[ii]+273.15,71.5+273.15).x[0]*100.
for w in ws:
    fig,ax=plt.subplots(1,1)
    scaled_t=(w-globt_s.values[-1])*slopes+hi_ta_p2 # New t
    td=ut._rhDew2d(nr,nc,hi_rh_p2,hi_rh_p2)+273.15 # Old rh
    scaled['%.0f'%w]=TaTd2HI(scaled_t+273.15,td,hiref,taref,tdref)-273.15 # new rh
    # frac20[wcount]=np.nansum(pop20['ssp3_2020'].data[scaled['%.0f'%w]>=71.5])/np.nansum(pop20).ssp3_2020.data
    # frac50[wcount]=np.nansum(pop50['ssp3_2050'].data[scaled['%.0f'%w]>=71.5])/np.nansum(pop50).ssp3_2050.data 
    counts,xbins,ybins,image=ax.hist2d(scaled_t.flatten(),hi_rh_p2.flatten(),bins=bins[wcount],\
                                          norm=colors.LogNorm(),cmap = "cividis")
    ax.plot(ta_pow,rh_lu,color='red',linestyle="-",linewidth=4)
    ax.set_xlabel("Air temperature [$^{\circ}$C]")
    ax.set_ylabel("Relative humidity [%]")
    ax.set_xlim(25,52)
    ax.set_ylim(0,100)
    ax.grid()
    fig.set_size_inches(6,5)
    fig.set_dpi(800)
    fig.savefig(plot_d+"ERA5_SCATTER_%.0f_+%.1fC.png"%(p2,w))
    
    wcount+=1
        
assert 1==2    

fig,ax=plt.subplots(1,1)
ax.plot(ta_pow,rh_lu,color='red',linestyle="-",linewidth=4)
ax.set_xlabel("Air temperature [$^{\circ}$C]")
ax.set_ylabel("Relative humidity [%]")
ax.grid()
ax.set_xlim(25,52)
ax.set_ylim(0,100)
fig.set_size_inches(6,5)
fig.set_dpi(800)
fig.savefig(plot_d+"WIRED_DEMO.png")
# vanos_old={}
# vanos_young={}
# vanos_death_hours_old=np.zeros((nr,nc))
# vanos_death_hours_young=np.zeros((nr,nc))

# for i in range(3,6):
#     vanos_old[i]=np.zeros((nt,nr,nc))
#     vanos_young[i]=np.zeros((nt,nr,nc))
    
#     for y in range(yst,ystp+1): 
#         vanos_old[i][y-yst,:,:]=pk.load(open(d+"vanos_old_%.0f_%.0f.p"%(i,y),'rb'))
#         vanos_young[i][y-yst,:,:]=pk.load(open(d+"vanos_young_%.0f_%.0f.p"%(i,y),'rb'))
        
#     vanos_death_hours_old+=np.sum(vanos_old[i],axis=0)/(ystp-yst+1)
#     vanos_death_hours_young+=np.sum(vanos_young[i],axis=0)/(ystp-yst+1)

# city["old_dead_hy"]= NN(city["lat"].values[:],city["lon"].values[:],\
#            lat2,lon2,vanos_death_hours_old)
# city["young_dead_hy"]= NN(city["lat"].values[:],city["lon"].values[:],\
#             lat2,lon2,vanos_death_hours_young)   
# city["death_risk_old"]=city["old_dead_hy"]*city["pop"]
# city["death_risk_young"]=city["young_dead_hy"]*city["pop"]
# city.sort_values(by="death_risk_old",ascending=False,inplace=True)
    
# vanos_death_hours_old[vanos_death_hours_old==0]=np.nan
# vanos_death_hours_young[vanos_death_hours_young==0]=np.nan

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
   
   
# Scatter 
fig,ax=plt.subplots(1,1,figsize=(5,4),dpi=500)
# Get cloud's upper limit
vals_p1,bins_p1,_ =binned_statistic(hi_ta_p1,hi_rh_p1,
                             statistic=lambda y: np.nanpercentile(y,99.99),
                             bins=25)

vals_p2,bins_p2,_ =binned_statistic(hi_ta_p2,hi_rh_p2,
                             statistic=lambda y: np.nanpercentile(y,99.99),
                             bins=25)

mids_p1=(bins_p1[1:]+bins_p1[0:-1])/2.
mids_p2=(bins_p2[1:]+bins_p2[0:-1])/2.
#counts,xbins,ybins,image = plt.hist2d(hi_ta_p1,hi_rh_p1,bins=100,\
                                      #norm=colors.LogNorm(),cmap = "turbo")

#ax.scatter(hi_ta,hi_rh,s=0.001,c='k')
ax.plot(mids_p1,vals_p1,color='magenta')
ax.plot(mids_p2,vals_p2,color='cyan')

#ax.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
#           levels=50)
#ax.plot(vl["ta"],rh_vl)
# ax.plot(ta_pow,rh_pow,color='k',linestyle="--",linewidth=3)
ax.plot(ta_pow,rh_lu,color='red',linestyle="-",linewidth=3)
ax.plot(ta_pow,rh_35,color='blue',linestyle="-",linewidth=3)
ax.plot()
ax.grid()
ax.set_xlabel("Ta [$^{\circ}$C]")
ax.set_ylabel("RH [%]")
ax.set_xlim(25,52)
ax.set_ylim(0,100)
fig.savefig(plot_d+"ERA5_SCATTER.png")

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




