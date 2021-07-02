
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
data_path = "/home/clairegayral/Documents/openclassroom/data/P4/"
res_path = "/home/clairegayral/Documents/openclassroom/res/P4/"


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
                   "SPD_micro_community",
                   "Seattle Police Department Micro Community Policing Plan Areas", # in [1,61]
                   "City Council Districts", # in [1,2]
                   "SPD Beats",# in [1,51]
                   "Zip Codes",# in [1,7]
                   "ZipCode",## in 2016 data
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
                'YearsENERGYSTARCertified','Comment','Comments'
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
                             "NonResidential","SPS-District K-12"],
                       } 

dict_cluster["PrimaryPropertyType"] = {
                        "medical":["Hospital", "Laboratory"],
                        "large/service":["Large Office","Supermarket / Grocery Store","Hotel",
                                         "University","High-Rise Multifamily",
                                         "Medical Office","Senior Care Community"],
                        "medium/other":["Restaurant","Other","Mixed Use Property",
                                        "Residence Hall","Retail Store","K-12 School",],
                        "medium":["Mid-Rise Multifamily","Refrigerated Warehouse",
                                  "Small- and Mid-Sized Office"],
                        "small/empty":["Non-Refrigerated Warehouse", "Distribution Center", "Warehouse",
                                       "Low-Rise Multifamily","Worship Facility"],
                        "small/other":["Self-Storage Facility","Office"],
                       } 


dict_cluster["LargestPropertyUseType"] = {
                        "hospital":['Hospital (General Medical & Surgical)'],
                        "reception":['Courthouse','Convention Center'],
                        "data_center":['Data Center'],
                        "side_medical":['Laboratory', 'Lifestyle Center',],
                        "1":['Police Station', 'Wholesale Club/Supercenter',
                                    'Urgent Care/Clinic/Other Outpatient', 'Senior Care Community',
                                    'Supermarket/Grocery Store', 'Hotel', 'Other - Restaurant/Bar',],
                        "2":['College/University', 'Other/Specialty Hospital', 'Restaurant',
                                         'Parking', 'Other - Mall', 'Fitness Center/Health Club/Gym',
                                         'Strip Mall', 'Medical Office', 'Museum',
                                         'Other - Entertainment/Public Assembly',],
                        "3":['Fire Station', 'Other','Other - Recreation',
                                          'Other - Public Services', 'Office',
               'Residence Hall/Dormitory', 'Other - Utility', 'Adult Education',
                                          'Financial Office', 'K-12 School', 'Other - Education',
                                          'Library', 'Other - Lodging/Residential', 'Retail Store', 
                                          'Social/Meeting Hall','Personal Services (Health/Beauty, Dry Cleaning, etc)',],
                        "4":['Manufacturing/Industrial Plant', 'Pre-school/Daycare',
               'Multifamily Housing', 'Performing Arts', 'Movie Theater',
               'Automobile Dealership', 'Other - Services', 'Refrigerated Warehouse'],
                        "5":['Repair Services (Vehicle, Shoe, Locksmith, etc)',
                            'Distribution Center', 'Non-Refrigerated Warehouse', 
                            'Worship Facility','Residential Care Facility', ],
                        "6":['Prison/Incarceration', 'Outpatient Rehabilitation/Physical Therapy', 'Bank Branch'],
                        "7":['Self-Storage Facility'],
                        "8":['Food Service'],
}


###################################

##
##
## select variables : 
## 
##

def select_columns(df, list_of_var):
    return(df[df.columns.intersection(list_of_var)])


