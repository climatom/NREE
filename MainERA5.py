#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:19:56 2023

@author: k2147389
"""

import time, cdsapi, utils as ut, pickle as pk, xarray as xa, numba as nb, \
    numpy as np
from scipy.interpolate import griddata


# Fast, explicit function to determine indices in lookup array
@nb.jit(nopython=True,fastmath=True)
def TaTd2HI(ta,td,hiref,taref,tdref):
    nt,nr,nc=ta.shape
    out=np.zeros((nt,nr,nc))*np.nan
    t_st=np.min(taref)
    td_st=np.min(tdref)
    res_t=taref[0,1]-taref[0,0]
    res_td=tdref[1,0]-tdref[0,0]
    for ti in nb.prange(nt):
        for row in nb.prange(nr):
            for col in nb.prange(nc):
                i=int(np.floor((ta[ti,row,col]-t_st)/res_t)) # Col
                j=int(np.floor((td[ti,row,col]-td_st)/res_td)) # Row
                
                # Might be that some cold obs fall off the lower end of the 
                # lookup table. Protect against that here. 
                if i < 0: i=0
                if j < 0: j=0
                
                # Note that ref's columns are ta; rows are td
                out[ti,row,col]=hiref[j,i]
                
    return out    

@nb.jit(nopython=True,fastmath=True)
def TaRhFlag(ta,rh,flag,taref,rhref):
    nt,nr,nc=ta.shape
    out=np.zeros((nt,nr,nc))*np.nan
    t_st=np.min(taref)
    rh_st=np.min(rhref)
    res_t=taref[0,1]-taref[0,0]
    res_rh=rhref[1,0]-rhref[0,0]
    maxrow,maxcol=taref.shape
    maxrow-=1
    maxcol-=1
    for ti in nb.prange(nt):
        for row in nb.prange(nr):
            for col in nb.prange(nc):
                
                i=int(np.floor((ta[ti,row,col]-t_st)/res_t)) # Col
                j=int(np.floor((rh[ti,row,col]-rh_st)/res_rh)) # Row
                
                # Might be that some cold obs fall off the lower end of the 
                # lookup table. Protect against that here. We can set to 1
                # in output (eq. to setting index to 0) because that means all is fine! 
                if i < 0: i=0
                if j < 0: j=0
                if i > maxcol: i=maxcol
                if j > maxrow:  j=maxrow
                               
                # Note that ref's columns are ta; rows are td
                out[ti,row,col]=flag[j,i] 

    return out

# Load the lookup tables
lookup_dir="/Users/k2147389/Desktop/Papers/homeostasis/output/"
# Resolution
res_rh=0.005
res_ta=0.05
# ../ta_lookup_ta%.3f_rh%.3f.p"%(res_ta,res_rh)
hiref=pk.load( open( lookup_dir+"HI_lookup_ta%.3f_rh%.3f.p"%(res_ta,res_rh), "rb" ) )
taref=pk.load( open( lookup_dir+"ta_lookup_ta%.3f_rh%.3f.p"%(res_ta,res_rh), "rb" ) )
tdref=pk.load( open( lookup_dir+"td_lookup_ta%.3f_rh%.3f.p"%(res_ta,res_rh), "rb" ) )
vanos_t=pk.load( open( lookup_dir+"vanos_t.p", "rb" ) )
vanos_rh=pk.load( open( lookup_dir+"vanos_rh.p", "rb" ) )
vanos_young=pk.load( open( lookup_dir+"vanos_young.p", "rb" ) )
vanos_old=pk.load( open( lookup_dir+"vanos_old.p", "rb" ) )
# Resolution



# Get ERA5
c = cdsapi.Client()
# Init the counter
counter=0
for y in range(2021,2023):
    for m in range(1,13):
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '2m_dewpoint_temperature', '2m_temperature', 
                    'surface_pressure',
                ],
                'year': '%.0f'%y,
                'month': '%02d'%m,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    65, -180, -65,
                    180,
                ],
                            },
            'download.nc')
        
        # Open the last download and load into arrays
        data=xa.open_dataset('download.nc')
        ta=data.t2m.data[:,:,:]
        td=data.d2m.data[:,:,:]
        p=data.sp.data[:,:,:]
        nt,nr,nc=ta.shape
            
        
        # Compute the HI (i.e., read from the lookup tables) 
        hi=TaTd2HI(ta,td,hiref,taref,tdref)
        
        
        # Compute the wet-bulb (Colin's formula)
        rh=ut._dewRH3D(nt,nr,nc,ta,td)
        tw=ut.WetBulb_all(ta.flatten()-273.15, p.flatten(), rh.flatten()*100.,\
                          Hum_mode=1).reshape(nt,nr,nc)

        
        
        # Evaluate the counts of the Vanos categories
        young=TaRhFlag(ta-273.15,rh*100.,vanos_young,vanos_t,vanos_rh)
        old=TaRhFlag(ta-273.15,rh*100.,vanos_old,vanos_t,vanos_rh)
        vanos_young_dict={}
        vanos_old_dict={}
        # Count how many hit 1-5
        for flag_i in [1,2,3,4,5]:
            
                vanos_young_dict["%.0f"%flag_i]=np.nansum(young==flag_i,axis=0)
                vanos_old_dict["%.0f"%flag_i]=np.nansum(old==flag_i,axis=0)
                
                if m >1:
                    _young=pk.load( open("../ERA5/vanos_young_%.0f_%.0f.p"%(flag_i,y),\
                                  "rb" ) )
                    _old=pk.load( open("../ERA5/vanos_old_%.0f_%.0f.p"%(flag_i,y),\
                                  "rb" ) )
                        
                    vanos_young_dict["%.0f"%flag_i]=vanos_young_dict["%.0f"%flag_i]+_young
                    vanos_old_dict["%.0f"%flag_i]=vanos_old_dict["%.0f"%flag_i]+_old
                
                # Write on every monthly step
                pk.dump(vanos_young_dict["%.0f"%flag_i],\
                        open( "../ERA5/vanos_young_%.0f_%.0f.p"%(flag_i,y), "wb" ) )
                pk.dump(vanos_old_dict["%.0f"%flag_i],\
                        open( "../ERA5/vanos_old_%.0f_%.0f.p"%(flag_i,y), "wb" ) )                    
                      

        #### Identify the top  and take relevant fields
        vn=["hi_","tw_","ta_"]
        vs=[hi,tw,ta]
        _dict={}
        _ii=0
        for v in vn:
            # In each iteration, we are taking the max values, in turn, of 
            # hi, tw, and ta. We then also take the  ta, p, and rh during 
            # these maxima 
            idx=np.nanargmax(vs[_ii],axis=0)
            _dict[v+"max"]=np.squeeze(np.take_along_axis(vs[_ii],\
                                                         idx[None,:,:],axis=0))
            _dict[v+"tamax"]=np.squeeze(np.take_along_axis(ta,idx[None,:,:],\
                                                           axis=0))
            _dict[v+"pmax"]=np.squeeze(np.take_along_axis(p,idx[None,:,:],\
                                                          axis=0))
            _dict[v+"rhmax"]=np.squeeze(np.take_along_axis(rh,idx[None,:,:],\
                                                           axis=0))
            maxi=np.nanmax(_dict[v+"max"])   
            if v != "tw_": maxi-=273.15
            print("Max %s for %.0f, %.0f is: %.2fC"%(v.strip("_"),y,m,maxi))
            
            if m >1:
                # Then we should read in the relevant data
                old_max=pk.load( open( "../ERA5/"+v+"max_%.0f.p"%y, "rb" ) )
                old_tamax=pk.load( open( "../ERA5/"+v+"tamax_%.0f.p"%y, "rb" ) )
                old_pmax=pk.load( open( "../ERA5/"+v+"pmax_%.0f.p"%y, "rb" ) )
                old_rhmax=pk.load( open( "../ERA5/"+v+"rhmax_%.0f.p"%y, "rb" ) )
                
                comp=np.array([_dict[v+"max"],old_max])
                comp_ta=np.array([_dict[v+"tamax"],old_tamax])
                comp_p=np.array([_dict[v+"pmax"],old_pmax])
                comp_rh=np.array([_dict[v+"rhmax"],old_rhmax])
                
                # Find max
                _idx_=np.nanargmax(comp,axis=0)
                
                # Update the maxima
                _dict[v+"max"]=np.squeeze(np.take_along_axis(comp,\
                                                    _idx_[None,:,:],axis=0))
                _dict[v+"tamax"]=np.squeeze(np.take_along_axis(comp_ta,\
                                                    _idx_[None,:,:],axis=0))
                _dict[v+"pmax"]=np.squeeze(np.take_along_axis(comp_p,\
                                                    _idx_[None,:,:],axis=0))
                _dict[v+"rhmax"]=np.squeeze(np.take_along_axis(comp_rh,\
                                                    _idx_[None,:,:],axis=0))
                         
            # Write on every monthly step
            pk.dump(_dict[v+"max"],open( "../ERA5/"+v+"max_%.0f.p"%y, "wb" ) )
            pk.dump(_dict[v+"tamax"],open( "../ERA5/"+v+"tamax_%.0f.p"%y, "wb" ) )
            pk.dump(_dict[v+"pmax"],open( "../ERA5/"+v+"pmax_%.0f.p"%y, "wb" ) )
            pk.dump(_dict[v+"rhmax"],open( "../ERA5/"+v+"rhmax_%.0f.p"%y, "wb" ) )   
            
            # CRITICAL!
            _ii+=1
            
            