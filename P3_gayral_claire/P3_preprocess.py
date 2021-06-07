
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn import preprocessing

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter

##
## META DATA ## 
##
data_path = "/home/clairegayral/Documents/openclassroom/data/P3/"
res_path = "/home/clairegayral/Documents/openclassroom/res/P3/"


numerical_var = ["NumberofFloors","PropertyGFATotal",
                 "PropertyGFAParking",'PropertyGFABuilding(s)',
                 'ENERGYSTARScore',
                 'SiteEUI(kBtu/sf)','SiteEUIWN(kBtu/sf)',
                 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)',
                 'SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)', 
                 'SteamUse(kBtu)','Electricity(kWh)',
                 'Electricity(kBtu)', 'NaturalGas(therms)',
                 'NaturalGas(kBtu)', 'OtherFuelUse(kBtu)',
                 'GHGEmissions(MetricTonsCO2e)', 'GHGEmissionsIntensity(kgCO2e/ft2)',
                 ## added var :
                 "CO2_emissions", "CO2_emissions_intensity",
                 "age_of_building",
                 ]

categorical_var = ["CouncilDistrictCode",# in [1,7]
                   "2010 Census Tracts", # in [116,135]
                   "Seattle Police Department Micro Community Policing Plan Areas", # in [1,61]
                   "City Council Districts", # in [1,2]
                   "SPD Beats",# in [1,51]
                   "Zip Codes",# in [1,7]
                   "BuildingType", # txt
                   "PrimaryPropertyType", # txt
                   "Neighborhood", # txt
                   'ComplianceStatus', # "Not Compliant","Compliant"
                   'LargestPropertyUseType'
                   ]

identification_var = ["OSEBuildingID", "DataYear","PropertyName",
                     "TaxParcelIdentificationNumber"]

unclassified = ["PropertyName", # txt
                "TaxParcelIdentificationNumber", # code for a spe tax
                "Location", # dict, hard to extract data
                "YearBuilt", # pass it into the age of the building in 2015
                'NumberofBuildings',
                ## I don't know how to treat these yet 
                "ListOfAllPropertyUseTypes"
                'LargestPropertyUseTypeGFA','SecondLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseTypeGFA'
                'LargestPropertyUseType', 'SecondLargestPropertyUseType','ThirdLargestPropertyUseType',
                'YearsENERGYSTARCertified','Comment',
                'DefaultData', # "yes","no"
                'Outlier', # 'High Outlier', 'Low Outlier'
                ]


#####################################




###################################

##
##
## select variables : 
## 
##

def select_columns(df, list_of_var):
    return(df[df.columns.intersection(list_of_var)])


