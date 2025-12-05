def llenar_cubo(lado):
    """Llena un cubo 3D - O(n^3)"""
    cubo = []
    for i in range(lado):
        matriz = []
        for j in range(lado):
            fila = []
            for k in range(lado):
                fila.append(i * 100 + j * 10 + k)
            matriz.append(fila)
        cubo.append(matriz)
    return cubo
