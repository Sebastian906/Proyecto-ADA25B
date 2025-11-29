import os
from core.patterns import PatternRecognizer

def test_divide_and_conquer():
    code = """
    function mergeSort(arr, inicio, fin)
        if inicio < fin then
            medio = (inicio + fin) / 2
            mergeSort(arr, inicio, medio)
            mergeSort(arr, medio + 1, fin)
            merge(arr, inicio, medio, fin)
        end
    end
    """
    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())


def test_dynamic_programming():
    code = """
    function knapsack(weights, values, capacity)
        n = len(weights)
        dp = array[n+1][capacity+1]

        for i = 0 to n do
            for w = 0 to capacity do
                if i == 0 or w == 0 then
                    dp[i][w] = 0
                else if weights[i-1] <= w then
                    dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                else
                    dp[i][w] = dp[i-1][w]
                end
            end
        end
        return dp[n][capacity]
    end
    """
    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())


def test_greedy_algorithm():
    code = """
    function activitySelection(activities)
        sort(activities by finish time)
        selected = []
        selected.append(activities[0])
        for i = 1 to len(activities) do
            if activities[i].start >= selected[-1].finish then
                selected.append(activities[i])
            end
        end
        return selected
    end
    """
    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())


def test_backtracking():
    code = """
    function solveNQueens(board, col)
        if col >= N then
            printSolution(board)
            return true
        end

        for i = 0 to N-1 do
            if isSafe(board, i, col) then
                board[i][col] = 1
                if solveNQueens(board, col + 1) then
                    return true
                end
                board[i][col] = 0
            end
        end
        return false
    end
    """
    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())


def test_branch_and_bound():
    code = """
    function branchAndBoundKnapsack(items, capacity)
        queue = []
        queue.append(rootNode)
        maxProfit = 0

        while queue is not empty do
            node = queue.pop()
            if node.bound > maxProfit then
                if node.isLeaf == false then
                    queue.append(leftChild(node))
                    queue.append(rightChild(node))
                end
                if node.profit > maxProfit then
                    maxProfit = node.profit
                end
            end
        end
        return maxProfit
    end
    """
    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())


def test_graph_algorithms():
    code = """
    function bfs(graph, start)
        queue = []
        visited = set()
        queue.append(start)
        visited.add(start)

        while queue is not empty do
            node = queue.pop(0)
            for neighbor in graph[node] do
                if neighbor not in visited then
                    queue.append(neighbor)
                    visited.add(neighbor)
                end
            end
        end
    end
    """
    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())


def test_sorting_algorithm():
    code = """
    function bubbleSort(arr)
        for i = 0 to len(arr) - 1 do
            for j = 0 to len(arr) - i - 1 do
                if arr[j] > arr[j + 1] then
                    swap(arr[j], arr[j + 1])
                end
            end
        end
    end
    """
    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())


def analyze_pseudocode_files():
    """
    Analiza los archivos de pseudocódigo en formato .txt dentro de la carpeta tests/pseudocode.
    """
    folder_path = "tests/pseudocode"
    if not os.path.exists(folder_path):
        print(f"La carpeta '{folder_path}' no existe.")
        return

    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not files:
        print(f"No se encontraron archivos .txt en la carpeta '{folder_path}'.")
        return

    print(f"\nAnalizando archivos en '{folder_path}':")
    for file in files:
        file_path = os.path.join(folder_path, file)
        print(f"\nArchivo: {file}")
        print("-" * 50)
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            recognizer = PatternRecognizer()
            patterns = recognizer.analyze(code)
            print(recognizer.get_summary())
        print("-" * 50)


def analyze_test_patterns():
    """
    Ejecuta las pruebas definidas en el archivo test_patterns.py.
    """
    print("Test 1: Divide y Conquista")
    test_divide_and_conquer()
    print("\nTest 2: Programación Dinámica")
    test_dynamic_programming()
    print("\nTest 3: Algoritmos Voraces")
    test_greedy_algorithm()
    print("\nTest 4: Backtracking")
    test_backtracking()
    print("\nTest 5: Branch and Bound")
    test_branch_and_bound()
    print("\nTest 6: Algoritmos de Grafos")
    test_graph_algorithms()
    print("\nTest 7: Algoritmos de Ordenamiento")
    test_sorting_algorithm()


def insert_algorithm():
    """
    Permite al usuario insertar un algoritmo en pseudocódigo directamente desde la terminal.
    """
    print("\nInserte su algoritmo en pseudocódigo. Escriba 'FIN' en una nueva línea para terminar:")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "FIN":
            break
        lines.append(line)
    
    code = "\n".join(lines)
    print("\nAlgoritmo ingresado:")
    print("-" * 50)
    print(code)
    print("-" * 50)

    recognizer = PatternRecognizer()
    patterns = recognizer.analyze(code)
    print(recognizer.get_summary())


def main():
    """
    Menú principal para elegir entre analizar test_patterns, pseudocódigos en .txt, o insertar un algoritmo.
    """
    while True:
        print("\nSeleccione una opción:")
        print("1. Analizar algoritmos en test_patterns.py")
        print("2. Analizar pseudocódigos en la carpeta tests/pseudocode")
        print("3. Insertar un algoritmo en pseudocódigo")
        print("4. Salir")
        choice = input("Ingrese su elección (1/2/3/4): ").strip()

        if choice == "1":
            analyze_test_patterns()
        elif choice == "2":
            analyze_pseudocode_files()
        elif choice == "3":
            insert_algorithm()
        elif choice == "4":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()