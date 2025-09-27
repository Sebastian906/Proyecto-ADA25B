"""
Parser de Pseudocódigo para Análisis de Complejidad
Utiliza Lark para parsear estructuras algorítmicas comunes
"""

from lark import Lark, Transformer, Tree, Token, v_args
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
        self.complexity_hint = None  # Para almacenar pistas de complejidad
    
    def add_child(self, child: 'ASTNode'):
        self.children.append(child)
    
    def __repr__(self):
        return f"ASTNode({self.type}, {self.value}, {len(self.children)} children)"
    
class PseudocodeTransformer(Transformer):
    """
    Transforma el árbol del parsing de Lark a nuestro AST personalizado
    """
    @v_args(inline=True)
    def program(self, *statements):
        return ASTNode("program", children=list(statements))
    
    @v_args(inline=True)
    def statement(self, stmt):
        return stmt
    
    @v_args(inline=True)
    def assignment(self, var, expr):
        node = ASTNode("assignment", value=var)
        node.add_child(expr)
        return node
    
    @v_args(inline=True)
    def for_loop(self, var, start, end, body):
        node = ASTNode("for_loop", value={
            "variable": var,
            "start": start,
            "end": end
        })
        node.add_child(body)
        return node
    
    @v_args(inline=True)
    def while_loop(self, condition, body):
        node = ASTNode("while_loop")
        node.add_child(condition)
        node.add_child(body)
        return node
    
    @v_args(inline=True)
    def if_statement(self, condition, then_body, else_body=None):
        node = ASTNode("if_statement")
        node.add_child(condition)
        node.add_child(then_body)
        if else_body:
            node.add_child(else_body)
        return node
    
    @v_args(inline=True)
    def function_def(self, name, params, body):
        node = ASTNode("function", value={
            "name": name,
            "params": params
        })
        node.add_child(body)
        return node
    
    @v_args(inline=True)
    def function_call(self, name, args):
        return ASTNode("function_call", value={
            "name": name,
            "args": args
        })
    
    @v_args(inline=True)
    def block(self, *statements):
        return ASTNode("block", children=list(statements))
    
    @v_args(inline=True)
    def expression(self, expr):
        return expr
    
    @v_args(inline=True)
    def binary_op(self, left, op, right):
        node = ASTNode("binary_op", value=str(op))
        node.add_child(left)
        node.add_child(right)
        return node
    
    @v_args(inline=True)
    def array_access(self, array, index):
        node = ASTNode("array_access", value=array)
        node.add_child(index)
        return node
    
    @v_args(inline=True)
    def variable(self, name):
        return ASTNode("variable", value=str(name))
    
    @v_args(inline=True)
    def number(self, n):
        return ASTNode("number", value=int(n))
    
    @v_args(inline=True)
    def identifier(self, name):
        return str(name)
    
class PseudocodeParser:
    def __init__(self):
        # Gramática simplificada para bucles y condicionales
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

    ejemplos = [
        ("Búsqueda Lineal", """
            for i = 1 to n do
                x = x + 1
            end
        """),
        ("Condicional", """
            if x == y then
                x = x + 1
            end
        """),
        ("Bucle While", """
            while i < n do
                i = i + 1
            end
        """)
    ]

    for nombre, codigo in ejemplos:
        print(f"=== Ejemplo: {nombre} ===")
        try:
            ast = parser.parse(codigo)
            info = parser.extract_structure_info(ast)
            print("AST:", ast.pretty())
            print("Estructura:", info)
        except Exception as e:
            print("Error:", e)