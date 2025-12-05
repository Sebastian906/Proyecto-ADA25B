def suma_simple(n):
    """Suma los números del 1 al n - O(n)"""
    total = 0
    for i in range(n):
        total = total + i
    return total
