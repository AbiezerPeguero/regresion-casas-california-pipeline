
def filtrar_outliers_estructurales(X, y):
    """
    Elimina filas con topes artificiales de recoleccion de datos
    (precio == 500001, edad == 52) tanto de X como de y.
    Regla fija: no aprende nada, funciona igual en train y test.

    Recibe: X (DataFrame de features), y (Serie del target)
    Devuelve: X_filtrado, y_filtrado (mismo indice entre ambos)
    """
    filas_validas = (y != 500001) & (X['housing_median_age'] != 52)
    X_filtrado = X[filas_validas]
    y_filtrado = y[filas_validas]
    return X_filtrado, y_filtrado
