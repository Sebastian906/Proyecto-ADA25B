def busqueda_binaria(arr, x):
    """Búsqueda binaria iterativa - O(log n)"""
    izquierda = 0
    derecha = len(arr) - 1
    
    while izquierda <= derecha:
        medio = (izquierda + derecha) // 2
        
        if arr[medio] == x:
            return medio
        
        if arr[medio] < x:
            izquierda = medio + 1
        else:
            derecha = medio - 1
    
    return -1
