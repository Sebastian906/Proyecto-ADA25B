"""
Conversor de Python a Pseudocódigo para análisis de complejidad
"""

import ast
import re
from typing import List, Dict


class PythonToPseudocodeConverter:
    """
    Convierte código Python a pseudocódigo compatible con ComplexityAnalyzer
    """
    
    def __init__(self):
        self.indent_level = 0
        self.pseudocode_lines = []
    
    def convert(self, python_code: str) -> str:
        """
        Convierte código Python a pseudocódigo
        
        Args:
            python_code: Código Python válido
            
        Returns:
            Pseudocódigo equivalente
        """
        try:
            tree = ast.parse(python_code)
            self.pseudocode_lines = []
            self.indent_level = 0
            
            for node in tree.body:
                self._convert_node(node)
            
            return "\n".join(self.pseudocode_lines)
        
        except SyntaxError as e:
            print(f"Error de sintaxis en Python: {e}")
            return self._fallback_conversion(python_code)
    
    def _convert_node(self, node: ast.AST):
        """Convierte un nodo AST a pseudocódigo"""
        
        if isinstance(node, ast.FunctionDef):
            self._convert_function(node)
        
        elif isinstance(node, ast.For):
            self._convert_for(node)
        
        elif isinstance(node, ast.While):
            self._convert_while(node)
        
        elif isinstance(node, ast.If):
            self._convert_if(node)
        
        elif isinstance(node, ast.Assign):
            self._convert_assignment(node)
        
        elif isinstance(node, ast.Return):
            self._convert_return(node)
        
        elif isinstance(node, ast.Expr):
            # Expresiones standalone (como print)
            if isinstance(node.value, ast.Call):
                self._convert_call(node.value, standalone=True)
    
    def _convert_function(self, node: ast.FunctionDef):
        """function nombre(args)"""
        args = ", ".join(arg.arg for arg in node.args.args)
        self._add_line(f"function {node.name}({args})")
        
        self.indent_level += 1
        for child in node.body:
            self._convert_node(child)
        self.indent_level -= 1
        
        self._add_line("end")
    
    def _convert_for(self, node: ast.For):
        """for i = start to end do"""
        # Detectar range()
        target = self._get_var_name(node.target)
        
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == 'range':
                args = node.iter.args
                
                if len(args) == 1:
                    # range(n)
                    end = self._get_expr_str(args[0])
                    self._add_line(f"for {target} = 0 to {end}-1 do")
                
                elif len(args) == 2:
                    # range(start, end)
                    start = self._get_expr_str(args[0])
                    end = self._get_expr_str(args[1])
                    self._add_line(f"for {target} = {start} to {end}-1 do")
                
                else:
                    # range(start, end, step)
                    self._add_line(f"for {target} = 0 to n do")
            else:
                # for x in collection
                collection = self._get_expr_str(node.iter)
                self._add_line(f"for {target} in {collection} do")
        else:
            self._add_line(f"for {target} in collection do")
        
        self.indent_level += 1
        for child in node.body:
            self._convert_node(child)
        self.indent_level -= 1
        
        self._add_line("end")
    
    def _convert_while(self, node: ast.While):
        """while condition do"""
        condition = self._get_expr_str(node.test)
        self._add_line(f"while {condition} do")
        
        self.indent_level += 1
        for child in node.body:
            self._convert_node(child)
        self.indent_level -= 1
        
        self._add_line("end")
    
    def _convert_if(self, node: ast.If):
        """if condition then ... else ... end"""
        condition = self._get_expr_str(node.test)
        self._add_line(f"if {condition} then")
        
        self.indent_level += 1
        for child in node.body:
            self._convert_node(child)
        self.indent_level -= 1
        
        if node.orelse:
            self._add_line("else")
            self.indent_level += 1
            for child in node.orelse:
                self._convert_node(child)
            self.indent_level -= 1
        
        self._add_line("end")
    
    def _convert_assignment(self, node: ast.Assign):
        """variable = expresion"""
        targets = ", ".join(self._get_var_name(t) for t in node.targets)
        value = self._get_expr_str(node.value)
        self._add_line(f"{targets} = {value}")
    
    def _convert_return(self, node: ast.Return):
        """return value"""
        if node.value:
            value = self._get_expr_str(node.value)
            self._add_line(f"return {value}")
        else:
            self._add_line("return")
    
    def _convert_call(self, node: ast.Call, standalone=False):
        """Llamada a función"""
        func_name = self._get_expr_str(node.func)
        args = ", ".join(self._get_expr_str(arg) for arg in node.args)
        
        if standalone:
            self._add_line(f"{func_name}({args})")
        
        return f"{func_name}({args})"
    
    def _get_var_name(self, node: ast.AST) -> str:
        """Obtiene nombre de variable"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            # arr[i]
            return self._get_expr_str(node)
        else:
            return "var"
    
    def _get_expr_str(self, node: ast.AST) -> str:
        """Convierte expresión a string"""
        if isinstance(node, ast.Name):
            return node.id
        
        elif isinstance(node, ast.Constant):
            return str(node.value)
        
        elif isinstance(node, ast.Num):  # Python < 3.8
            return str(node.n)
        
        elif isinstance(node, ast.BinOp):
            left = self._get_expr_str(node.left)
            right = self._get_expr_str(node.right)
            op = self._get_op_str(node.op)
            return f"{left} {op} {right}"
        
        elif isinstance(node, ast.Compare):
            left = self._get_expr_str(node.left)
            ops = [self._get_compare_str(op) for op in node.ops]
            comparators = [self._get_expr_str(comp) for comp in node.comparators]
            
            result = left
            for op, comp in zip(ops, comparators):
                result += f" {op} {comp}"
            return result
        
        elif isinstance(node, ast.Call):
            return self._convert_call(node)
        
        elif isinstance(node, ast.Subscript):
            value = self._get_expr_str(node.value)
            index = self._get_expr_str(node.slice)
            return f"{value}[{index}]"
        
        elif isinstance(node, ast.Attribute):
            value = self._get_expr_str(node.value)
            return f"{value}.{node.attr}"
        
        elif isinstance(node, ast.List):
            elements = [self._get_expr_str(e) for e in node.elts]
            return f"[{', '.join(elements)}]"
        
        else:
            return "expr"
    
    def _get_op_str(self, op: ast.operator) -> str:
        """Convierte operador a string"""
        ops = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**'
        }
        return ops.get(type(op), '?')
    
    def _get_compare_str(self, op: ast.cmpop) -> str:
        """Convierte comparador a string"""
        ops = {
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>=',
            ast.Is: 'is',
            ast.IsNot: 'is not',
            ast.In: 'in',
            ast.NotIn: 'not in'
        }
        return ops.get(type(op), '?')
    
    def _add_line(self, line: str):
        """Agrega línea con indentación"""
        indent = "    " * self.indent_level
        self.pseudocode_lines.append(indent + line)
    
    def _fallback_conversion(self, python_code: str) -> str:
        """
        Conversión básica cuando falla el parsing AST
        Usa expresiones regulares simples
        """
        lines = python_code.split('\n')
        pseudo_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # def function -> function
            if line.startswith('def '):
                line = re.sub(r'def\s+(\w+)\s*\((.*?)\):', r'function \1(\2)', line)
            
            # for ... in range() -> for ... = start to end
            elif 'for ' in line and ' in range(' in line:
                match = re.search(r'for\s+(\w+)\s+in\s+range\((.*?)\):', line)
                if match:
                    var = match.group(1)
                    args = match.group(2).split(',')
                    
                    if len(args) == 1:
                        line = f"for {var} = 0 to {args[0].strip()}-1 do"
                    elif len(args) == 2:
                        line = f"for {var} = {args[0].strip()} to {args[1].strip()}-1 do"
            
            # while ... : -> while ... do
            elif line.startswith('while '):
                line = re.sub(r'while\s+(.*?):', r'while \1 do', line)
            
            # if ... : -> if ... then
            elif line.startswith('if '):
                line = re.sub(r'if\s+(.*?):', r'if \1 then', line)
            
            # elif -> else if
            elif line.startswith('elif '):
                line = re.sub(r'elif\s+(.*?):', r'else if \1 then', line)
            
            # else: -> else
            elif line.startswith('else:'):
                line = "else"
            
            # return
            elif line.startswith('return '):
                pass  # Mantener igual
            
            # Reemplazar operadores
            line = line.replace('//', '/').replace('and', 'AND').replace('or', 'OR')
            
            pseudo_lines.append(line)
        
        return "\n".join(pseudo_lines)


# === FUNCIÓN DE UTILIDAD ===

def convert_python_to_pseudocode(python_code: str) -> str:
    """
    Función helper para convertir Python a pseudocódigo
    
    Args:
        python_code: Código Python válido
        
    Returns:
        Pseudocódigo equivalente
        
    Ejemplo:
        >>> python_code = '''
        ... def fibonacci(n):
        ...     if n <= 1:
        ...         return n
        ...     return fibonacci(n-1) + fibonacci(n-2)
        ... '''
        >>> pseudo = convert_python_to_pseudocode(python_code)
        >>> print(pseudo)
        function fibonacci(n)
            if n <= 1 then
                return n
            end
            return fibonacci(n - 1) + fibonacci(n - 2)
        end
    """
    converter = PythonToPseudocodeConverter()
    return converter.convert(python_code)


# === TESTS ===

if __name__ == "__main__":
    # Test 1: Búsqueda binaria
    python_code1 = """
def busqueda_binaria(arr, x):
    izquierda = 0
    derecha = len(arr) - 1
    
    while izquierda <= derecha:
        medio = (izquierda + derecha) // 2
        
        if arr[medio] == x:
            return medio
        
        if arr[medio] < x:
            izquierda = medio + 1
        else:
            derecha = medio - 1
    
    return -1
"""
    
    print("TEST 1: Búsqueda Binaria")
    print("="*50)
    pseudo1 = convert_python_to_pseudocode(python_code1)
    print(pseudo1)
    print()
    
    # Test 2: Fibonacci
    python_code2 = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    print("TEST 2: Fibonacci")
    print("="*50)
    pseudo2 = convert_python_to_pseudocode(python_code2)
    print(pseudo2)
    print()
    
    # Test 3: Loops anidados
    python_code3 = """
for i in range(n):
    for j in range(n):
        if matriz[i][j] > 0:
            suma = suma + matriz[i][j]
"""
    
    print("TEST 3: Loops Anidados")
    print("="*50)
    pseudo3 = convert_python_to_pseudocode(python_code3)
    print(pseudo3)