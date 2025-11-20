from core.patterns import PatternRecognizer

def test_specific_patterns():
    code = """
    function divide_y_conquista(arr, inicio, fin)
        if inicio < fin then
            medio = (inicio + fin) / 2
            mergeSort(arr, inicio, medio)
            mergeSort(arr, medio + 1, fin)
            merge(arr, inicio, medio, fin)
        end
    end

    function binarySearch(arr, target)
        left = 0
        right = len(arr) - 1
        while left <= right do
            middle = (left + right) // 2
            if arr[middle] == target then
                return middle
            else if arr[middle] < target then
                left = middle + 1
            else
                right = middle - 1
            end
        end
        return -1
    end
    """

    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())

if __name__ == "__main__":
    test_specific_patterns()