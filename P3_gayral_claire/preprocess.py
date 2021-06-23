
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
##
## Clustering on modalities by semantic : 
##

dict_cluster = {}
dict_cluster["BuildingType"] = {
                        "campus":["Campus"],
                        "HR":["Multifamily HR (10+)"],
                        "ML":["Multifamily MR (5-9)"],
                        "LR":["Multifamily LR (1-4)"],
                        "other":["Nonresidential WA","Nonresidential COS",
                             "Nonresidential","SPS-District K-12"],
                       } 
dict_cluster["PrimaryPropertyType"] = {
                        "medical":["Hospital", "Laboratory"],
                        "large":["Large Office","Supermarket / Grocery Store"],
                        "service":["Hotel","University","High-Rise Multifamily",
                             "Medical Office","Senior Care Community",
                             "Restaurant"],
                        "other":["Other","Mixed Use Property","Retail Store", "Residence Hall"],
                        "medium":["Mid-Rise Multifamily","Refrigerated Warehouse","K-12 School","Small- and Mid-Sized Office"],
                        "empty":["Non-Refrigerated Warehouse", "Distribution Center","Warehouse" ],
                        "small":["Low-Rise Multifamily"",Worship Facility"],
                        "other2":["Self-Storage Facility","Office"],
                       } 


###################################




##
##
## select variables : 
## 
##

def select_columns(df, list_of_var):
    return(df[df.columns.intersection(list_of_var)])


