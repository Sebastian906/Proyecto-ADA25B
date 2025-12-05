def contar_digitos(n):
    """Cuenta los dígitos de un número - O(log n)"""
    contador = 0
    while n > 0:
        contador = contador + 1
        n = n // 10
    return contador
