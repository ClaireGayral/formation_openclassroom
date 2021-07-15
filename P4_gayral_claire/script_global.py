##
## product 
##

dimension_vars = ['product_weight_g', 'product_length_cm',       
                  'product_height_cm', 'product_width_cm']
description_vars = ['product_name_lenght', 'product_description_lenght',
                    'product_photos_qty']
                    
def get_coeffs_project2d_products(products): 
    
    X_std = get_X_missing_vals_imputed(products, std = True)
    
    X_dimension = X_std.loc[:,dimension_vars]
    X_description = X_std.loc[:,description_vars]

    my_pca = PCA(n_components=1)
    my_pca.fit(X_dimension)
    dimension_coeffs = pd.Series(my_pca.components_[0], index = dimension_vars)
    my_pca.fit(X_description)
    description_coeffs = pd.Series(my_pca.components_[0], index = description_vars)
    return(dimension_coeffs, description_coeffs)
