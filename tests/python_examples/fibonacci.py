def fibonacci(n):
    """Calcula fibonacci recursivamente - O(2^n)"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
