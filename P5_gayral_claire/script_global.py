##
## product 
##

dimension_vars = ['product_weight_g', 'product_length_cm',       
                  'product_height_cm', 'product_width_cm']
description_vars = ['product_name_lenght', 'product_description_lenght',
                    'product_photos_qty']
                    
def get_X_missing_vals_imputed(products, std = False):
    ## extract numeric table with id as index
    X = products.copy()
    X = X.set_index("product_id")
    X = X.loc[:,~(X.dtypes == object)]
    ## Standardize 
    my_std = StandardScaler().fit(X)
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
                  
##TODO : coeffs to be revaluated every year, otherwize, use pre-computed coeffs
def project2d_products(products, coeffs, var) :
    dimension_coeffs, description_coeffs = coeffs
    dimension_vars, description_vars = var
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
    res = pd.merge(res["product_category_name"], product_dimension, left_index=True, right_index=True)
    res = pd.merge(res, product_dimension, left_index=True, right_index=True)
    return(res)
