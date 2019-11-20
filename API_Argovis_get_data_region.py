# made for Python 3.  It may work with Python 2.7, but has not been well tested

# libraries to call for all python API calls on Argovis

import requests
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from datetime import datetime
import pdb
import os
from netCDF4 import Dataset as netcdf_dataset
from datetime import datetime, timedelta

#####



# Get a selected region from Argovis 

def get_selection_profiles(startDate, endDate, shape, presRange=None):
    baseURL = 'https://argovis.colorado.edu/selection/profiles/'
    startDateQuery = '?startDate=' + startDate
    endDateQuery = '&endDate=' + endDate
    shapeQuery = '&shape='+shape
    if not presRange == None:
        pressRangeQuery = '&presRange;=' + presRange
        url = baseURL + startDateQuery + endDateQuery + pressRangeQuery + shapeQuery
        # print('url accessed: ', url)
    else:
        url = baseURL + startDateQuery + endDateQuery + shapeQuery
    resp = requests.get(url)
    # Consider any status other than 2xx an error
    if not resp.status_code // 100 == 2:
        return "Error: Unexpected response {}".format(resp)
    selectionProfiles = resp.json()
    return selectionProfiles

## Get platform information
def parse_into_df(profiles,df):
    #initialize dict
    # meas_keys = profiles[0]['measurements'][0].keys()
    # df = pd.DataFrame(columns=meas_keys)
    for profile in profiles:
        if len(profile['measurements']) > 50: # don't include profiles that are short
            profileDf = pd.DataFrame(profile['measurements'])
            profileDf['cycle_number'] = profile['cycle_number']
            profileDf['profile_id'] = profile['_id']
            profileDf['lat'] = profile['lat']
            profileDf['lon'] = profile['lon']
            profileDf['date'] = profile['date']
            df = pd.concat([df, profileDf], sort=False)
    return df

# set start date, end date, lat/lon coordinates for the shape of region and pres range

# startDate='2019-7-12'
# endDate='2019-7-13'


# shape should be nested array with lon, lat coords.
# minlat = int(-50)
# maxlat = int(50)

# minlon = int(130)
# maxlonN = int(-125)
# maxlonS = int(-70) # create polygon to avoid Gulf floats

# shape = f'[[[{minlon},{minlat}],[{minlon},{maxlat}],[{maxlonN},{maxlat}],[{maxlonS},{minlat}],[{minlon},{minlat}]]]'

shape = f'[[[{130},{-50}],[{115},{13}],[{120},{50}],[{-125},{50}],[{-99},{18.5}],[{-66.5},{0}],[{-73},{-50}],[{130},{-50}]]]'

# # oldshape = '[[[-18.6,31.7],[-18.6,37.7],[-5.9,37.7],[-5.9,31.7],[-18.6,31.7]]]'
# print(shape)

presRange='[0,1000]'
# Get current directory to save file into
baseDir = '/Users/eaton/argo'
# os.getcwd()

earliest = datetime(2018,6,1).date()

for ii in range(490):
    startDate = (earliest + timedelta(days = ii))
    endDate = (startDate + timedelta(days = 1))
    startDate = str(startDate)
    endDate = str(endDate)
    print(startDate,endDate)


    selectionProfiles = get_selection_profiles(startDate, endDate, shape, presRange)

    if len(selectionProfiles) > 0:
        # print('parsing json')
        try:
            selectionDf = parse_into_df(selectionProfiles,selectionDf)
        except NameError:
            meas_keys = selectionProfiles[0]['measurements'][0].keys()
            selectionDf = pd.DataFrame(columns=meas_keys)
            selectionDf = parse_into_df(selectionProfiles,selectionDf)

    else:
        print('Returned empty')
    savefilename = os.path.join(baseDir,'region_'+startDate+'.csv')
    selectionDf.to_csv(savefilename)
    del selectionDf



