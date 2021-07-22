import pandas as pd
import numpy as np


data_path = "/home/clairegayral/Documents/openclassroom/data/P4/"
res_path = "/home/clairegayral/Documents/openclassroom/res/P4/"

from sklearn import preprocessing
from sklearn.impute import KNNImputer

###################
#### open data ####
###################

product_category_name_translation = pd.read_csv(data_path 
                        + "product_category_name_translation.csv")
sellers = pd.read_csv(data_path + "olist_sellers_dataset.csv")
products = pd.read_csv(data_path + "olist_products_dataset.csv")
orders = pd.read_csv(data_path + "olist_orders_dataset.csv")
order_reviews = pd.read_csv(data_path + "olist_order_reviews_dataset.csv")
order_payments = pd.read_csv(data_path + "olist_order_payments_dataset.csv")
order_items = pd.read_csv(data_path + "olist_order_items_dataset.csv")
geolocation = pd.read_csv(data_path + "olist_geolocation_dataset.csv")
customers = pd.read_csv(data_path + "olist_customers_dataset.csv")


## Lien entre les tables :
## order-product
link_order_product = pd.merge(orders["order_id"], 
    order_items[["order_id","product_id"]], 
    on = "order_id", how = 'right')
link_order_product


## customer-order
link_customer_order = pd.merge(customers[["customer_unique_id","customer_id"]], 
    orders[["customer_id","order_id"]], 
    on = "customer_id", how = 'right')


##########################
#### Construction RFM ####
##########################

##
## Recency
##
tmp = pd.merge(customers[["customer_id","customer_unique_id"]], 
               orders[["customer_id", "order_id","order_purchase_timestamp"]], 
               on="customer_id", how="right")
## get the lastest order date of each customer 
customer_last_timestamp = tmp[["customer_unique_id",
           "order_purchase_timestamp"]].groupby("customer_unique_id").max()
## use datetime format
customer_last_timestamp = pd.to_datetime(customer_last_timestamp["order_purchase_timestamp"],
                     format = "%Y-%m-%d %H:%M:%S")
## substrack the date of the latest command in the data : 
t_max = customer_last_timestamp.max()
recency = pd.Series(t_max-customer_last_timestamp, name = "recency")
## get the difference in decimal days format : 
recency =  recency / np.timedelta64(1, "D")
recency = recency.reset_index()

rfm = recency

##
## Frequency
##

frequency = tmp.customer_unique_id.value_counts()
frequency = pd.Series(frequency).reset_index()
frequency = frequency.rename(columns={"index":"customer_unique_id",
                                      "customer_unique_id":"frequency"})
rfm = pd.merge(rfm, frequency, on="customer_unique_id", how="left")

##
## Monetary Value
##

tmp = pd.merge(tmp, order_payments[["order_id","payment_value"]], 
               on="order_id", how="left")
monetary_value = tmp.groupby("customer_unique_id").sum()
monetary_value = monetary_value.reset_index()
monetary_value = monetary_value.rename(columns={"payment_value":"monetary_value"})
rfm = pd.merge(rfm, monetary_value, on="customer_unique_id", how="left")
rfm = rfm.set_index("customer_unique_id")

########################################
#### construction table my_products ####
########################################

##
## Product items : 
##
def get_X_missing_vals_imputed(products, std = False):
    ## extract numeric table with id as index
    X = products.copy()
    X = X.set_index("product_id")
    X = X.loc[:,~(X.dtypes == object)]
    ## Standardize 
    my_std = preprocessing.StandardScaler().fit(X)
    X_std = pd.DataFrame(my_std.transform(X), columns=X.columns, index=X.index)
    n_neighbors = 10 
    ## Impute missing values :
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_std = imputer.fit_transform(X_std)
    X_std = pd.DataFrame(X_std, index = X.index, columns = X.columns)
    if std :
        return(X_std)
    else :
        ## Inverse-standardize : 
        X = my_std.inverse_transform(X_std)
        X = pd.DataFrame(X, index = X_std.index, columns = X_std.columns)
        return(X)
def project2d_products(products, coeffs, var) :
    dimension_vars, description_vars = var
    dimension_coeffs = coeffs[dimension_vars]
    description_coeffs = coeffs[description_vars]
    X_std = get_X_missing_vals_imputed(products, std = True)
    ## project on coeffs
    product_dimension = np.dot(X_std.loc[:,dimension_vars],dimension_coeffs)
    product_dimension = pd.Series(product_dimension, name="product_dimension",
                                 index = X_std.index)
    product_description = np.dot(X_std.loc[:,description_vars],description_coeffs)
    product_description = pd.Series(product_description, name="product_description",
                                 index = X_std.index)
    ## concat on index = product_id
    res = products.copy()
    res = res.set_index("product_id")
#     res = pd.merge(res["product_category_name"], product_dimens/ion, left_index=True, right_index=True)
    res = pd.merge(product_dimension, product_description, left_index=True, right_index=True)
    return(res)

dimension_vars = ['product_weight_g', 'product_length_cm',       
                  'product_height_cm', 'product_width_cm']
description_vars = ['product_name_lenght', 'product_description_lenght',
                    'product_photos_qty']
coeffs = pd.read_csv(res_path+"products_items_coeffs_PCA.csv",index_col=0, squeeze=True)
my_products = project2d_products(products, coeffs=coeffs,
                                 var=(dimension_vars, description_vars))
my_products = my_products.reset_index()

##
## Ordered product
##

## descriptive statistic : how often it is ordered ? 
product_nb_app = order_items.product_id.value_counts()
my_quantile_classes = {"very_low":[1,0.95], "low":[0.95,0.75],
                       "medium_low" : [0.75,0.5],"medium_high" : [0.5,0.25],
                       "high" : [0.25,0.05], "very_high" : [0.05,0]}
my_products.loc[:,"product_freq_buy"] = 0
for freq_class in my_quantile_classes.keys():
    q_max, q_min = my_quantile_classes[freq_class] 
    n_max = product_nb_app.quantile(q=q_max)
    n_min = product_nb_app.quantile(q=q_min)
    if ~(n_min == n_max) : 
        cond_max = np.where(product_nb_app<=n_max)
        cond_min = np.where(product_nb_app>n_min)
        if q_max == 1 :
            range_index = cond_max
        elif q_min == 0 :
            range_index = cond_min
        else :
            range_index = np.intersect1d(cond_min,cond_max)
        product_in_class = order_items.product_id.iloc[range_index].values
        product_class_index = my_products.loc[my_products["product_id"].isin(product_in_class),:].index
        my_products.at[product_class_index,"product_freq_buy"] = np.mean([q_min,q_max])        
## ordered alone
my_products.loc[:,"product_flag_ordered_alone"] = 0
range_index = np.where(order_items.product_id.value_counts()==1)
products_ordered_alone = order_items.product_id.iloc[range_index].values
products_ordered_alone = my_products.index.isin(products_ordered_alone)
my_products.at[products_ordered_alone,"product_flag_ordered_alone"] = 1
## prices
tmp = order_items.groupby("product_id").mean()[['price', 'freight_value']]
tmp = tmp.add_prefix("product_")
tmp = tmp.reset_index()
my_products = pd.merge(my_products,tmp, on="product_id",how="left")

##
## Product category
##

# ## old version :
# y = products[["product_id","product_category_name"]]
# y = y.set_index("product_id").astype("object")
# y = y.fillna("missing")
# y = y.astype("category")

# all_cat = pd.Series(y["product_category_name"].cat.categories, name ="product_category_old" )
# new_cat = pd.Series(y["product_category_name"].cat.categories, name ="product_category_name" )
# for idx in all_cat.index :
#     cat = all_cat[idx]
#     new_cat.at[idx] = cat.split("_")[0]
# rename_cat = pd.merge(all_cat,new_cat, left_index=True, right_index=True)
# rename_cat = rename_cat.set_index("product_category_old").to_dict()
# rename_cat['la']="cuisine"
# y = y.replace(rename_cat["product_category_name"])
# y = y.astype("category").reset_index()
# my_products = pd.merge(y,my_products, on="product_id",how="left")

rename_categories_english = {
    "home_furnitures" : ['bed_bath_table','furniture_decor', 
                         'housewares','office_furniture',
                         'kitchen_dining_laundry_garden_furniture',
                         'home_confort','furniture_mattress_and_upholstery',
                         'furniture_living_room', 'furniture_bedroom',
                         'home_comfort_2',
                        ],
    "home_electronics":['small_appliances','air_conditioning',
                        'home_appliances','home_appliances_2',
                        'la_cuisine','small_appliances_home_oven_and_coffee',
                        "kitchen_portables_and_food_preparers"
                       ],
    "electronics":['computers_accessories','telephony',
                   'tablets_printing_image', 'fixed_telephony',
                   'consoles_games', 'audio','electronics',
                   'computers',
                  ],
    "multimedia" : ['books_general_interest','books_imported',
                    'cine_photo','music', 
                    'cds_dvds_musicals', 'dvds_blu_ray',
                   ], 
    "fashion" : ['fashion_bags_accessories','fashion_shoes',
                 'fashion_male_clothing','fashion_underwear_beach',
                 'fashion_sport', 'fashio_female_clothing',
                ],
    "children" : [ 'baby','toys',
                   'fashion_childrens_clothes'
                 ],
    "health" : ['health_beauty', 'perfumery',
                'diapers_and_hygiene'
               ],
    "food_drink" : ['food_drink','market_place',
                    'agro_industry_and_commerce','food',
                    'drinks'
                    ],
    "leisure" : ['auto','sports_leisure',
                 'watches_gifts',  'stationery',
                 'luggage_accessories', 'pet_shop',
                 'party_supplies','musical_instruments',
                 'arts_and_craftmanship',
                ],
    "decoration" : ['cool_stuff','art',
                    'christmas_supplies','flowers',
                   ],
    "DIY" : ['garden_tools','construction_tools_construction',
             'costruction_tools_garden','costruction_tools_tools', 
             'books_technical','home_construction',
             'construction_tools_lights','construction_tools_safety',
             'industry_commerce_and_business',
            ],
    "security" : ['signaling_and_security','security_and_services']
    }

rename_cat = product_category_name_translation.copy()
## manquait 2 variables dans la table de traduction :
rename_cat = rename_cat.append({"product_category_name":"portateis_cozinha_e_preparadores_de_alimentos",
                   "product_category_name_english" : "kitchen_portables_and_food_preparers"},
                   ignore_index=True)
rename_cat = rename_cat.append({"product_category_name": "pc_gamer",
                   "product_category_name_english" : "pc_gamer"},
                   ignore_index=True)
for new_cat, list_old_cat in rename_categories_english.items():
    bool_idx = rename_cat["product_category_name_english"].isin(list_old_cat)
    cat_idx = rename_cat.loc[bool_idx].index
    rename_cat.at[cat_idx, "new_cat_english"] = new_cat
dict_rename_cat = rename_cat[["product_category_name","new_cat_english"]]
dict_rename_cat = dict_rename_cat.set_index("product_category_name")
dict_rename_cat = dict_rename_cat.to_dict()["new_cat_english"]

y = products[["product_id","product_category_name"]]
# y = y.set_index("product_id")
y = y.replace(dict_rename_cat)
y = y.astype("category")

if "product_category_name" in my_products.columns:
    my_products = my_products.drop(columns="product_category_name")
my_products = pd.merge(y,my_products, on="product_id",how="right")

## set product_id as index : 
my_products = my_products.set_index("product_id")


######################################
#### construction table my_orders ####
######################################

##
## order_status
##

my_orders = orders[['order_id', 'customer_id']].copy()
y = orders[['order_id','order_status']]
y = y.set_index("order_id").astype("category")
my_orders = pd.merge(my_orders, y, on="order_id", how="left")

##
## Time and dates
##

order_dates = pd.DataFrame(index=orders.order_id)
## Purchase date and time split : 
purchase_timestamp = pd.to_datetime(
        orders.order_purchase_timestamp, format="%Y-%m-%d %H:%M:%S")
purchase_date = pd.to_datetime(
        purchase_timestamp.dt.date, format="%Y-%m-%d")
order_dates.at[:,"order_purchase_date"] = purchase_date.values
purch_time = purchase_timestamp.dt.time 
order_dates.at[:,"order_purchase_time"] = purch_time.values
order_dates = order_dates.astype({"order_purchase_date":"object",
                                  "order_purchase_time":"object"})
## Delta time for delivery date comparison (ctm = customer)
estim_delivery_date = pd.to_datetime(
        orders.order_estimated_delivery_date, format="%Y-%m-%d %H:%M:%S")
delivered_ctm_date = pd.to_datetime(
        orders.order_delivered_customer_date, format="%Y-%m-%d %H:%M:%S") 
delivered_carrier_date = pd.to_datetime(
        orders.order_delivered_carrier_date, format="%Y-%m-%d %H:%M:%S") 
delta_estim_declared = estim_delivery_date - delivered_ctm_date
delta_estim_declared = delta_estim_declared / np.timedelta64(1, "D")
order_dates.at[:,"order_dt_estim_declared"] = delta_estim_declared.values
delta_ctm_carrier = delivered_ctm_date - delivered_carrier_date
delta_ctm_carrier = delta_ctm_carrier / np.timedelta64(1, "D")
order_dates.at[:,"order_dt_ctm_carrier"] = delta_ctm_carrier.values
my_orders = pd.merge(my_orders, order_dates, on="order_id",how="left")

##
## Prices 
##

tmp = order_items.groupby("order_id").sum()
tmp = tmp.reset_index().drop(columns="order_item_id")
my_orders = pd.merge(my_orders, tmp, on="order_id",how="left")

##
## Payments
##

## payment_installments :
tmp = order_payments.sort_values("payment_installments",ascending=False)
tmp = tmp.drop_duplicates("order_id","first")
my_orders = pd.merge(my_orders, tmp[["order_id","payment_installments"]], 
                     on="order_id", how="left")
## most of payment type (max sum of value): 
# extract sum of payment value for each payment type : 
sub_table = order_payments[["order_id","payment_value","payment_type"]]
sub_table = sub_table.sort_values("order_id")
index = sub_table.drop_duplicates(["order_id","payment_type"]).index
tmp = sub_table.groupby(["order_id","payment_type"]).sum()
tmp = tmp.reset_index().set_index(index)
## drop duplicates where smaller payment_values (sum)
tmp = tmp.sort_values("payment_value", ascending=False)
tmp = tmp.drop_duplicates("order_id", keep="first")
payment_type = tmp.copy()
y = tmp[['order_id','payment_type']]
y = y.astype({"payment_type":"category"}) 
my_orders = pd.merge(my_orders, y, on="order_id", how="left")
## number of payment type : 
tmp = order_payments[["order_id","payment_type"]].drop_duplicates()
tmp = tmp["order_id"].value_counts().reset_index()
tmp = tmp.rename(columns={"index":"order_id", "order_id":"nb_payment_type"})
my_orders = pd.merge(my_orders, tmp, on="order_id", how="left")

##
## Reviews 
##

## add count_reviews, count_messages, count_title, for one order : 
tmp = order_reviews.groupby("order_id").count().reset_index()
tmp = tmp[["order_id","review_id", "review_comment_title", "review_comment_message"]]
# renaming columns ("count" instead of "review")
dict_rename = {"review_id":"count_review"}
for colname in tmp.columns[2:] :
    dict_rename[colname] = colname.replace("review","count")
tmp = tmp.rename(columns=dict_rename)
my_orders = pd.merge(my_orders, tmp, on="order_id", how="left")

##
## Feature issues des variables numériques de produits
##
tmp = pd.merge(link_order_product, my_products, 
               on="product_id", how="left")
tmp = tmp.drop(columns=['product_id',"product_flag_ordered_alone",
                        "product_price", "product_freight_value"])
tmp = tmp.groupby("order_id").sum()
tmp = tmp.add_prefix("sum_")
tmp = tmp.reset_index()
my_orders = pd.merge(my_orders,tmp, on="order_id", how="left")

##
## Feature issues des categories de produits
##

## extract dummies 
product_cat_dummies = pd.get_dummies(my_products["product_category_name"])
product_cat_dummies = product_cat_dummies.add_prefix("product_category_")
product_cat_dummies = product_cat_dummies.set_index(products["product_id"]).reset_index()
## get nb of categories in the same product
order_cat = pd.merge(link_order_product,product_cat_dummies, on="product_id", how="left")
order_diff_cat = order_cat.drop(columns="product_id")
order_diff_cat = order_diff_cat.groupby("order_id").sum()
order_diff_cat[order_diff_cat>1] = 1
tmp = order_diff_cat.sum(axis=1)
tmp = tmp.reset_index().rename(columns={0:"count_prod_cat"})
my_orders = pd.merge(my_orders, tmp, on="order_id", how="left")
## get nb max of product in the same category
order_cat_no_duplc = order_cat.drop_duplicates(subset=["order_id","product_id"])
order_cat_no_duplc = order_cat_no_duplc.drop(columns="product_id")
order_cat_no_duplc = order_cat.groupby("order_id").sum()
order_cat_no_duplc.sum(axis=0)
tmp = order_cat_no_duplc.max(axis=1)
tmp = tmp.rename("count_max_product_in_cat")


## to compute one hot encoder, finally keep categorical for the moment
# enc = OneHotEncoder(handle_unknown='ignore')
# enc.fit(y)
# y_dummies = pd.DataFrame(enc.transform(y).toarray(),
#                           index = y.index, columns=np.unique(y))
# y_dummies = y_dummies.add_prefix("order_status_").reset_index()
# y_dummies = y_dummies.astype("category")
# my_orders = pd.merge(my_orders, y_dummies, on="order_id", how="left")

## set order_id as index and drop customer_id : 
my_orders = my_orders.drop(columns="customer_id")
my_orders = my_orders.set_index("order_id")

#########################################
#### Construction table my_customers ####
#########################################

tmp = customers.drop_duplicates(subset="customer_unique_id")
my_customers = tmp[["customer_unique_id","customer_zip_code_prefix"]]

my_customers = pd.merge(my_customers, rfm, on="customer_unique_id", how="left")

