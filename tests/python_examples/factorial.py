def factorial(n):
    """Calcula factorial de n recursivamente - O(n)"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
