def buscar_potencia(n):
    """Encuentra la potencia de 2 más cercana - O(log n)"""
    potencia = 1
    exponente = 0
    while potencia < n:
        potencia = potencia * 2
        exponente = exponente + 1
    return exponente
