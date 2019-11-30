# made for Python 3.  It may work with Python 2.7, but has not been well tested

# libraries to call for all python API calls on Argovis

import numpy as np
import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import pickle
# from netCDF4 import Dataset as netcdf_dataset
# from scipy.interpolate import griddata
# from datetime import datetime
# import pdb

def get_url(startDate, endDate, shape, presRange=None):
    """Returns the URL that will be sent to the Argo API"""
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
    return url


def get_selection_profiles(url):
    """Pings the API with a given url and returns the data in .json format"""
    resp = requests.get(url)
    # Consider any status other than 2xx an error
    if not resp.status_code // 100 == 2:
        return "Error: Unexpected response {}".format(resp)
    selectionProfiles = resp.json()
    return selectionProfiles


def parse_into_df(profiles,df):
    """Parses the .json data to a pandas dataframe"""
    for profile in profiles:
        if len(profile['measurements']) > 50: # don't include profiles that are short
            profileDf = pd.DataFrame(profile['measurements'])
            profileDf['profile_id'] = profile['_id']
            profileDf['lat'] = profile['lat']
            profileDf['lon'] = profile['lon']
            profileDf['date'] = profile['date']
            df = pd.concat([df, profileDf], sort=False)
    df = df[['profile_id', 'pres', 'temp', 'lat', 'lon', 'psal', 'date']]
    df.fillna(value='NULL', inplace=True)
    return df


def check_pkl(picklename):
    """Checks if there is already the .pkl file in the current directory"""
    if os.path.exists(picklename):
        pkl_bool = True
        print(f'"{picklename}" file found')
    else:
        pkl_bool = False
        print(f'No "{picklename}" file found')
    return pkl_bool


# Instantiate variables
earliest = datetime(2018,6,1).date()
startDate = (earliest + timedelta(days=0))
endDate = (startDate + timedelta(days=1))
startDate = str(startDate)
endDate = str(endDate)
shape = f'[[[{130},{-50}],[{115},{13}],[{120},{50}],[{-125},{50}],' \
        f'[{-99},{18.5}],[{-66.5},{0}],[{-73},{-50}],[{130},{-50}]]]'
presRange='[0,1000]'
baseDir = os.getcwd()
savefilename = os.path.join(baseDir, 'argo_data' + '.csv')
if not os.path.exists(savefilename):
    with open(savefilename, 'w'):
        pass
picklename = os.path.join(baseDir, 'argo_urls' + '.pkl')
pkl_bool = check_pkl(picklename)
current_urls = set()
bool_val = True

# Read in the .pkl file if it exists
if pkl_bool:
    with open(picklename, "rb") as input_file:
        print(f'Loading previous URLs from: "{picklename}"')
        past_urls = pickle.load(input_file)

# Main loop at appends data to .csv and seen urls to .pkl file
for ii in range(490):
# for ii in range(2):

    startDate = (earliest + timedelta(days = ii))
    endDate = (startDate + timedelta(days = 1))
    startDate = str(startDate)
    endDate = str(endDate)

    url = get_url(startDate, endDate, shape, presRange)
    # print(url)
    short_url = url.split('presRange')[0][0:-1]
    current_urls.add(short_url)

    if pkl_bool:
        if short_url in past_urls:
            continue

    selectionProfiles = get_selection_profiles(url)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('----------------------------------------------------------------')
    print(startDate,endDate)
    print(short_url)
    print(f'Done at {current_time}')

    if len(selectionProfiles) > 0:
        # print('parsing json')
        try:
            selectionDf = parse_into_df(selectionProfiles,selectionDf)
        except NameError:
            meas_keys = selectionProfiles[0]['measurements'][0].keys()
            selectionDf = pd.DataFrame(columns=meas_keys)
            selectionDf = parse_into_df(selectionProfiles, selectionDf)
    else:
        print('Returned empty')

    selectionDf.to_csv(savefilename, mode='a', header=bool_val, index=None)
    bool_val = False
    del selectionDf

    # Either write the seen urls to a new .pkl file or append to an existing one
    if not pkl_bool:
        with open(picklename, "wb") as output_file:
            pickle.dump(current_urls, output_file)
    if pkl_bool:
        if current_urls != past_urls:
            all_urls = past_urls.union(current_urls)
            with open(picklename, "wb") as output_file:
                pickle.dump(all_urls, output_file)

print('----------------------------------------------------------------')
if pkl_bool:
    if current_urls == past_urls:
        print('No new URLs were found')
