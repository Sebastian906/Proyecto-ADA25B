"""
Parser de Pseudocódigo para Análisis de Complejidad (versión extendida línea a línea)
Utiliza Lark para parsear estructuras algorítmicas comunes y permite
analizar ejecución línea por línea para calcular propiedades detalladas.
"""

from lark import Lark, Transformer, Tree, Token
from typing import Dict, List, Any
import re

class ASTNode:
    """
    Nodo base del Árbol de sintaxis abstracta
    """
    def __init__(self, type_: str, value: Any = None, children: List['ASTNode'] = None):
        self.type = type_
        self.value = value
        self.children = children or []

        # Propiedades nuevas para análisis detallado
        self.line_no = None
        self.raw = None
        self.variables_read: List[str] = []
        self.variables_written: List[str] = []
        self.exec_count = None
        self.time_cost = None
        self.space_cost = None

    def add_child(self, child: 'ASTNode'):
        self.children.append(child)

    def __repr__(self):
        return f"ASTNode({self.type}, {self.value}, {len(self.children)} children)"
    
class PseudocodeTransformer(Transformer):
    @staticmethod
    def make_node(type_, *children, value=None):
        node = ASTNode(type_, value=value)
        for c in children:
            if isinstance(c, ASTNode):
                node.add_child(c)
        return node

    def program(self, *statements):
        return ASTNode("program", children=list(statements))

    def statement(self, stmt):
        return stmt

    def assignment(self, var, expr):
        node = ASTNode("assignment", value=str(var))
        node.add_child(expr)
        return node

    def for_loop(self, var, start, end, body):
        node = ASTNode("for_loop", value={
            "variable": str(var),
            "start": str(start),
            "end": str(end)
        })
        node.add_child(body)
        return node

    def while_loop(self, condition, body):
        node = ASTNode("while_loop", value=str(condition))
        node.add_child(body)
        return node

    def if_statement(self, condition, then_body, else_body=None):
        node = ASTNode("if_statement", value=str(condition))
        node.add_child(then_body)
        if else_body:
            node.add_child(else_body)
        return node

    def block(self, *statements):
        return ASTNode("block", children=list(statements))

    def expression(self, expr):
        return expr

    def variable(self, name):
        return ASTNode("variable", value=str(name))

    def number(self, n):
        return ASTNode("number", value=int(n))
    
class PseudocodeParser:
    def __init__(self):
        self.grammar = r"""
            start: statement+
            statement: for_loop
                        | while_loop
                        | if_statement
                        | assignment

            for_loop: "for" NAME "=" NUMBER "to" NAME "do" statement+ "end"
            while_loop: "while" NAME "<" NAME "do" statement+ "end"
            if_statement: "if" NAME "==" NAME "then" statement+ "end"
            assignment: NAME "=" NAME ("+"|"-") NUMBER

            NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
            NUMBER: /[0-9]+/

            %import common.WS
            %ignore WS
        """

        self.parser = Lark(self.grammar, start="start")

    def parse(self, code: str) -> Tree:
        """Parses pseudocode into an AST (Tree)."""
        return self.parser.parse(code)
    
    def normalize_spanish(self, code: str) -> str:
        translations = [
            (r"PARA (\\w+) DESDE (\\d+) HASTA (\\w+) HACER", r"for \1 = \2 to \3 do"),
            (r"FIN PARA", "end"),
            (r"FIN", "end"),
            (r"MIENTRAS (.+?) HACER", r"while \1 do"),
            (r"SI (.+?) ENTONCES", r"if \1 then"),
            (r"ESCRIBIR", "write"),
            (r"LEER", "read"),
            (r"<-", "=")
        ]
        for pat, rep in translations:
            code = re.sub(pat, rep, code, flags=re.IGNORECASE)
        return code
    
    def analyze_by_line(self, code: str) -> List[Dict[str, Any]]:
        lines = code.strip().splitlines()
        results = []
        recurrence = "T(n) = O(1)"  # valor por defecto

        for i, line in enumerate(lines, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            normalized = self.normalize_spanish(raw_line)

            node = ASTNode("statement", value=normalized)
            node.line_no = i
            node.raw = raw_line

            if re.match(r"for", normalized, re.IGNORECASE):
                node.type = "for_loop"
                node.exec_count = "n"
                node.time_cost = "O(n)"
            elif re.match(r"while", normalized, re.IGNORECASE):
                node.type = "while_loop"
                node.exec_count = "log n" if "izquierda" in normalized.lower() and "derecha" in normalized.lower() else "k"
                node.time_cost = "O(log n)" if node.exec_count == "log n" else "O(k)"
            elif re.match(r"if", normalized, re.IGNORECASE):
                node.type = "if_statement"
                node.exec_count = "1"
                node.time_cost = "O(1)"
            elif re.match(r"write", normalized, re.IGNORECASE):
                node.type = "write"
                node.exec_count = "1"
                node.time_cost = "O(1)"
            elif re.match(r"read", normalized, re.IGNORECASE):
                node.type = "read"
                node.exec_count = "1"
                node.time_cost = "O(1)"
            elif re.match(r"\w+\s*=", normalized):
                node.type = "assignment"
                node.exec_count = "1"
                node.time_cost = "O(1)"
            else:
                node.type = "other"

            tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", normalized)
            node.variables_read = [t for t in tokens if t.lower() not in
                                    ["for", "to", "do", "end", "if", "then", "while", "write", "read"]]
            node.variables_written = [tokens[0]] if "=" in normalized else []

            if "arreglo" in normalized.lower():
                node.space_cost = "O(n)"

            results.append({
                "line": i,
                "raw": raw_line,
                "type": node.type,
                "variables_read": node.variables_read,
                "variables_written": node.variables_written,
                "exec_count": node.exec_count,
                "time_cost": node.time_cost,
                "space_cost": node.space_cost
            })

        # -----------------------------------------------
        # Cálculo de ecuación de recurrencia
        # -----------------------------------------------
        has_recursion = any("function_call" in r["type"] for r in results)
        has_for = any(r["type"] == "for_loop" for r in results)
        has_while = any(r["type"] == "while_loop" for r in results)

        if has_recursion:
            recurrence = "T(n) = a·T(n/b) + O(f(n))"
        elif has_for:
            recurrence = "T(n) = T(n-1) + O(1)"
        elif has_while:
            # Si detectamos una búsqueda binaria por variables clave:
            binary_pattern = any(
                "izquierda" in r["raw"].lower() and "derecha" in r["raw"].lower()
                for r in results
            )
            if binary_pattern:
                recurrence = "T(n) = T(n/2) + O(1)"
            else:
                recurrence = "T(n) = T(n-1) + O(1)"
        else:
            recurrence = "T(n) = O(1)"

        # Agregar la ecuación final como línea extra de salida
        results.append({
            "line": len(results) + 1,
            "raw": "// Ecuación de recurrencia global",
            "type": "summary",
            "variables_read": [],
            "variables_written": [],
            "exec_count": None,
            "time_cost": None,
            "space_cost": None,
            "recurrence": recurrence
        })

        return results

    def extract_structure_info(self, ast: Tree):
        """Recorre el AST y extrae bucles y condicionales."""
        info = {"loops": [], "conditionals": []}

        def visit(node):
            if isinstance(node, Tree):
                if node.data == "for_loop":
                    info["loops"].append({"type": "for", "node": node})
                elif node.data == "while_loop":
                    info["loops"].append({"type": "while", "node": node})
                elif node.data == "if_statement":
                    info["conditionals"].append({"type": "if", "node": node})

                # Recursivamente visitar los hijos
                for child in node.children:
                    visit(child)

            elif isinstance(node, Token):
                # Si quieres capturar variables, números, etc., lo haces aquí
                pass

        visit(ast)
        return info


if __name__ == "__main__":
    parser = PseudocodeParser()

    code = """
    INICIO
    ESCRIBIR "Ingrese el tamaño del arreglo ordenado:"
    LEER n
    DEFINIR arreglo COMO ENTERO[n]

    PARA i DESDE 0 HASTA n-1 HACER
        arreglo[i] <- i + 1
    FIN PARA

    ESCRIBIR "Ingrese el número a buscar:"
    LEER x

    izquierda <- 0
    derecha <- n - 1
    encontrado <- FALSO

    MIENTRAS izquierda <= derecha HACER
        medio <- (izquierda + derecha) / 2
        SI arreglo[medio] == x ENTONCES
            encontrado <- VERDADERO
            derecha <- izquierda - 1
        SINO
            SI arreglo[medio] < x ENTONCES
                izquierda <- medio + 1
            SINO
                derecha <- medio - 1
            FIN SI
        FIN SI
    FIN MIENTRAS

    SI encontrado == VERDADERO ENTONCES
        ESCRIBIR "Elemento encontrado."
    SINO
        ESCRIBIR "Elemento no encontrado."
    FIN SI
    FIN
    """

    analysis = parser.analyze_by_line(code)
    for line_info in analysis:
        if line_info["type"] != "summary":  # no mostrar la línea de resumen aquí
            print(f"Línea {line_info['line']}: {line_info['raw']}")
            print(f"  Tipo: {line_info['type']}")
            print(f"  Variables leídas: {line_info['variables_read']}")
            print(f"  Variables escritas: {line_info['variables_written']}")
            print(f"  Ejecuciones: {line_info['exec_count']}")
            print(f"  Costo temporal: {line_info['time_cost']}")
            print(f"  Costo espacial: {line_info['space_cost']}")
            print()

    summary = next((item for item in analysis if item["type"] == "summary"), None)
    if summary and "recurrence" in summary:
        print("Ecuación de recurrencia global detectada:")
        print(f"{summary['recurrence']}")