def buscar_en_matriz(matriz, valor):
    """Busca un valor en una matriz 2D - O(n^2)"""
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            if matriz[i][j] == valor:
                return (i, j)
    return None
