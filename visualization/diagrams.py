"""
Módulo `diagrams.py`

Este módulo se encarga de generar diagramas basados en el análisis de código o pseudocódigo. 
Incluye funcionalidades para construir Árboles de Sintaxis Abstracta (AST), detectar funciones recursivas, 
y exportar diagramas en múltiples formatos.

Clases principales:
    - DiagramNode: Representa un nodo en el diagrama.
    - ASTDiagramBuilder: Construye diagramas desde código Python o pseudocódigo.
    - GraphvizExporter: Exporta diagramas a formato Graphviz (DOT y SVG).
    - MermaidExporter: Exporta diagramas a formato Mermaid (Markdown).
    - NetworkxExporter: Exporta diagramas a formato PNG usando NetworkX.
    - RecursionTreeVisualizer: Genera representaciones de árboles de recursión.
    - JsonExporter: Exporta diagramas a formato JSON.

Funciones principales:
    - analyze_and_visualize: Función principal para analizar código y generar diagramas.
"""

import ast
import json
import re
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path

# Importaciones del proyecto
from core.parser import PseudocodeParser
from core.complexity import ComplexityAnalyzer
from core.patterns import PatternRecognizer


@dataclass
class DiagramNode:
    """
    Representa un nodo en el diagrama.

    Atributos:
        node_id (str): Identificador único del nodo.
        node_type (str): Tipo del nodo (e.g., "for_loop", "assignment").
        label (str): Etiqueta del nodo.
        value (str): Valor asociado al nodo.
        children (List[str]): Lista de nodos hijos.
        line_number (Optional[int]): Número de línea en el código.
        complexity (Optional[str]): Complejidad estimada del nodo.
        pattern (Optional[str]): Patrón algorítmico detectado.
        is_recursive (bool): Indica si el nodo es parte de una función recursiva.
        recursion_depth (Optional[int]): Profundidad de recursión.
        exec_count (Optional[str]): Número de ejecuciones estimadas.
        time_cost (Optional[str]): Costo temporal estimado.
        space_cost (Optional[str]): Costo espacial estimado.
    """
    node_id: str
    node_type: str
    label: str
    value: str
    children: List[str] = field(default_factory=list)
    line_number: Optional[int] = None
    complexity: Optional[str] = None
    pattern: Optional[str] = None
    is_recursive: bool = False
    recursion_depth: Optional[int] = None
    exec_count: Optional[str] = None
    time_cost: Optional[str] = None
    space_cost: Optional[str] = None

def extract_algorithm_name(code: str) -> str: # type: ignore
        """Extrae el nombre del algoritmo del código"""
        # Intenta encontrar nombre de función
        match = re.search(r'def\s+(\w+)\s*\(|function\s+(\w+)|PROCEDIMIENTO\s+(\w+)', code, re.IGNORECASE)
        if match:
            name = match.group(1) or match.group(2) or match.group(3)
            return name.lower()
        
        # Si no hay función, intenta encontrar comentarios
        comment_match = re.search(r'#\s*(.+?)[\n$]', code)
        if comment_match:
            name = comment_match.group(1).strip()[:20].replace(' ', '_')
            return name.lower()
        
        # Fallback: genera nombre por timestamp
        from datetime import datetime
        return f"algorithm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class ASTDiagramBuilder:
    """
    Construye diagramas desde código Python o pseudocódigo.

    Métodos principales:
        - build_from_python_code(code: str) -> Optional[str]: Construye un AST desde código Python.
        - build_from_pseudocode(code: str) -> Dict[str, Any]: Construye un análisis desde pseudocódigo.
        - get_nodes() -> Dict[str, DiagramNode]: Retorna todos los nodos generados.
        - get_recursive_functions() -> Set[str]: Retorna funciones recursivas detectadas.
    """
    
    def __init__(self) -> None:
        self.nodes: Dict[str, DiagramNode] = {}
        self.node_counter: int = 0
        self.recursive_functions: Set[str] = set()
        self.parser: PseudocodeParser = PseudocodeParser()
        self.complexity_analyzer: ComplexityAnalyzer = ComplexityAnalyzer()
        self.pattern_recognizer: PatternRecognizer = PatternRecognizer()
    
    def _get_next_id(self) -> str:
        """Genera ID único para nodo"""
        self.node_counter += 1
        return f"n{self.node_counter}"
    
    
    def build_from_python_code(self, code: str) -> Optional[str]:
        """Construye AST desde código Python usando ast nativo"""
        self.nodes.clear()
        self.node_counter = 0
        self.recursive_functions.clear()
        
        try:
            tree = ast.parse(code)
            self._detect_recursion_python(tree)
            root_id = self._parse_python_node(tree, None, 0)
            return root_id
        except SyntaxError as e:
            print(f"Error de sintaxis: {e}")
            return None
    
    def build_from_pseudocode(self, code: str) -> Dict[str, Any]:
        """Construye análisis desde pseudocódigo usando PseudocodeParser"""
        self.nodes.clear()
        self.node_counter = 0
        
        try:
            # Analizar línea por línea
            line_analysis: List[Dict[str, Any]] = self.parser.analyze_by_line(code)
            
            # Crear nodos a partir del análisis
            for line_info in line_analysis:
                if line_info.get("type") == "summary":
                    continue
                
                node_id: str = self._get_next_id()
                node_type: str = str(line_info.get("type", "unknown"))
                raw_value: str = str(line_info.get("raw", ""))
                
                diagram_node = DiagramNode(
                    node_id=node_id,
                    node_type=node_type,
                    label=f"{node_type}: {raw_value[:30]}",
                    value=raw_value,
                    line_number=int(line_info.get("line", 0)) if line_info.get("line") else None,
                    exec_count=str(line_info.get("exec_count")) if line_info.get("exec_count") else None,
                    time_cost=str(line_info.get("time_cost")) if line_info.get("time_cost") else None,
                    space_cost=str(line_info.get("space_cost")) if line_info.get("space_cost") else None
                )
                
                self.nodes[node_id] = diagram_node
            
            # Análisis de complejidad
            complexity_data: Optional[Dict[str, Any]] = None
            try:
                complexity_result = self.complexity_analyzer.analyze(code)
                complexity_data = {
                    "big_o": complexity_result.big_o,
                    "omega": complexity_result.omega,
                    "theta": complexity_result.theta,
                    "explanation": complexity_result.explanation,
                    "recurrence": complexity_result.recurrence,
                    "pattern_type": complexity_result.pattern_type
                }
            except Exception as e:
                print(f"Advertencia en análisis de complejidad: {e}")
            
            # Obtener recurrencia
            recurrence_value: Optional[str] = None
            for item in line_analysis:
                if item.get("type") == "summary" and "recurrence" in item:
                    recurrence_value = item.get("recurrence")
                    break
            
            return {
                "nodes": self.nodes,
                "complexity": complexity_data,
                "recurrence": recurrence_value
            }
        
        except Exception as e:
            print(f"Error al procesar pseudocódigo: {e}")
            return {"nodes": {}, "complexity": None, "recurrence": None}
    
    def _detect_recursion_python(self, tree: ast.AST) -> None:
        """Detecta funciones recursivas en árbol Python"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            if child.func.id == node.name:
                                self.recursive_functions.add(node.name)
    
    def _extract_python_node_info(self, node: ast.AST) -> Tuple[str, str, str]:
        """Extrae información de un nodo Python AST"""
        node_type = node.__class__.__name__
        label = ""
        value = ""
        
        if isinstance(node, ast.FunctionDef):
            label = f"def {node.name}"
            value = f"{node.name}(...)"
        elif isinstance(node, ast.For):
            target = node.target.id if isinstance(node.target, ast.Name) else "var"
            label = "for loop"
            value = f"for {target} in ..."
        elif isinstance(node, ast.While):
            label = "while loop"
            value = "while condition"
        elif isinstance(node, ast.If):
            label = "if condition"
            value = "if ..."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                label = f"call: {node.func.id}()"
                value = node.func.id
            else:
                label = "function call"
                value = "call"
        elif isinstance(node, ast.Return):
            label = "return"
            value = "return"
        elif isinstance(node, ast.Assign):
            label = "assignment"
            value = "="
        elif isinstance(node, ast.Name):
            label = node.id
            value = node.id
        elif isinstance(node, ast.Constant):
            val_str = str(node.value)[:15]
            label = val_str
            value = val_str
        else:
            label = node_type
            value = node_type
        
        return node_type, label, value
    
    def _parse_python_node(self, node: ast.AST, parent_id: Optional[str], depth: int) -> str:
        """Parsea recursivamente un nodo Python AST"""
        node_id = self._get_next_id()
        node_type, label, value = self._extract_python_node_info(node)
        
        # Detectar recursión
        is_recursive = False
        recursion_depth = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in self.recursive_functions:
                is_recursive = True
                recursion_depth = depth
        
        # Procesar hijos
        children_ids: List[str] = []
        for child in ast.iter_child_nodes(node):
            child_id = self._parse_python_node(child, node_id, depth + 1)
            children_ids.append(child_id)
        
        # Asignar complejidad
        complexity = self._get_complexity_for_node_type(node_type)
        
        diagram_node = DiagramNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            value=value,
            children=children_ids,
            line_number=getattr(node, 'lineno', None),
            complexity=complexity,
            is_recursive=is_recursive,
            recursion_depth=recursion_depth
        )
        
        self.nodes[node_id] = diagram_node
        return node_id
    
    def _get_complexity_for_node_type(self, node_type: str) -> Optional[str]:
        """Asigna complejidad estimada por tipo"""
        complexity_map: Dict[str, str] = {
            'For': 'O(n)',
            'While': 'O(n)',
            'If': 'Ω(1)',
            'Call': 'Variable',
            'FunctionDef': 'Analysis',
        }
        return complexity_map.get(node_type)
    
    def get_nodes(self) -> Dict[str, DiagramNode]:
        """Retorna todos los nodos"""
        return self.nodes
    
    def get_recursive_functions(self) -> Set[str]:
        """Retorna funciones recursivas detectadas"""
        return self.recursive_functions


class GraphvizExporter:
    """Exporta diagramas a formato Graphviz (DOT)"""
    
    def __init__(self, builder: ASTDiagramBuilder) -> None:
        self.builder = builder
        self.dot_code: str = ""
    
    def generate(self) -> str:
        """Genera código DOT"""
        nodes = self.builder.get_nodes()
        
        self.dot_code = "digraph AST {\n"
        self.dot_code += "    rankdir=TB;\n"
        self.dot_code += "    node [shape=box, style=rounded, fontname=Arial, fontsize=10];\n"
        self.dot_code += "    graph [bgcolor=white, pad=0.5];\n\n"
        
        for node_id, node in nodes.items():
            color = self._get_color(node.node_type)
            label = node.label
            if node.complexity and node.complexity != 'Analysis':
                label += f"\\n{node.complexity}"
            
            self.dot_code += f'    {node_id} [label="{label}", fillcolor="{color}", style="rounded,filled"];\n'
        
        self.dot_code += "\n"
        
        for node_id, node in nodes.items():
            for child_id in node.children:
                self.dot_code += f"    {node_id} -> {child_id};\n"
        
        self.dot_code += "}\n"
        return self.dot_code
    
    def _get_color(self, node_type: str) -> str:
        """Asigna color por tipo de nodo"""
        colors: Dict[str, str] = {
            'For': '#FFB6C1',
            'While': '#FFB6C1',
            'If': '#87CEEB',
            'Call': '#98FB98',
            'FunctionDef': '#FFD700',
            'Return': '#FFA500',
            'Assign': '#DDA0DD',
            'for_loop': '#FFB6C1',
            'while_loop': '#FFB6C1',
            'if_statement': '#87CEEB',
        }
        return colors.get(node_type, '#D3D3D3')
    
    def save(self, filepath: str) -> None:
        """Guarda archivo DOT"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.dot_code)
        print(f"Archivo DOT guardado: {filepath}")
    
    def export_svg(self, dot_path: str, svg_path: str) -> bool:
        """Convierte DOT a SVG"""
        try:
            Path(svg_path).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ['dot', '-Tsvg', dot_path, '-o', svg_path],
                check=True,
                capture_output=True,
                timeout=10
            )
            print(f"SVG generado: {svg_path}")
            return True
        except Exception as e:
            print(f"No se pudo generar SVG: {e}")
            return False


class MermaidExporter:
    """Exporta diagramas a formato Mermaid"""
    
    def __init__(self, builder: ASTDiagramBuilder) -> None:
        self.builder = builder
        self.mermaid_code: str = ""
    
    def generate(self) -> str:
        """Genera código Mermaid"""
        nodes = self.builder.get_nodes()
        
        self.mermaid_code = "graph TD\n"
        
        for node_id, node in nodes.items():
            label = node.label[:40]
            if node.complexity and node.complexity != 'Analysis':
                label += f"<br/>{node.complexity}"
            
            node_class = self._get_class(node.node_type)
            self.mermaid_code += f'    {node_id}["{label}"] {node_class}\n'
        
        self.mermaid_code += "\n"
        
        for node_id, node in nodes.items():
            for child_id in node.children:
                self.mermaid_code += f"    {node_id} --> {child_id}\n"
        
        self.mermaid_code += self._get_styles()
        return self.mermaid_code
    
    def _get_class(self, node_type: str) -> str:
        """Clase Mermaid por tipo"""
        classes: Dict[str, str] = {
            'For': ':::loop',
            'While': ':::loop',
            'If': ':::condition',
            'Call': ':::call',
            'FunctionDef': ':::function',
            'Return': ':::return',
            'for_loop': ':::loop',
            'while_loop': ':::loop',
            'if_statement': ':::condition',
        }
        return classes.get(node_type, '')
    
    def _get_styles(self) -> str:
        """Define estilos Mermaid"""
        return """
    classDef loop fill:#FFB6C1,stroke:#333,stroke-width:2px,color:#000
    classDef condition fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    classDef call fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    classDef function fill:#FFD700,stroke:#333,stroke-width:2px,color:#000
    classDef return fill:#FFA500,stroke:#333,stroke-width:2px,color:#000
"""
    
    def save_markdown(self, filepath: str) -> None:
        """Guarda Markdown con Mermaid"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""# Diagrama AST - Análisis de Algoritmo

## Árbol de Sintaxis Abstracta

```mermaid
{self.mermaid_code}
```

## Leyenda de Colores

- Rosa: Bucles (for, while)
- Azul: Condicionales (if)
- Verde: Llamadas a funciones
- Oro: Definiciones
- Naranja: Retorno
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Markdown generado: {filepath}")


class NetworkxExporter:
    """Exporta diagramas con NetworkX (PNG)"""
    
    def __init__(self, builder: ASTDiagramBuilder) -> None:
        self.builder = builder
    
    def save_png(self, filepath: str) -> bool:
        """Genera PNG"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            nodes = self.builder.get_nodes()
            G = nx.DiGraph()
            
            for node_id, node in nodes.items():
                G.add_node(node_id, label=node.label)
            
            for node_id, node in nodes.items():
                for child_id in node.children:
                    G.add_edge(node_id, child_id)
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(16, 10))
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            color_map: List[str] = []
            for node_id in G.nodes():
                node_type: str = nodes[node_id].node_type
                if node_type in ['For', 'While', 'for_loop', 'while_loop']:
                    color_map.append('#FFB6C1')
                elif node_type in ['If', 'if_statement']:
                    color_map.append('#87CEEB')
                elif node_type == 'Call':
                    color_map.append('#98FB98')
                elif node_type == 'FunctionDef':
                    color_map.append('#FFD700')
                else:
                    color_map.append('#D3D3D3')
            
            nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2000, alpha=0.9)
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=1.5)
            
            labels: Dict[str, str] = {str(nid): str(nodes[nid].value[:10]) for nid in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title("AST - Arbol de Sintaxis Abstracta", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"PNG generado: {filepath}")
            return True
        except Exception as e:
            print(f"No se pudo generar PNG: {e}")
            return False


class RecursionTreeVisualizer:
    """Visualiza árboles de recursión"""
    
    def __init__(self, builder: ASTDiagramBuilder) -> None:
        self.builder = builder
    
    def generate_tree(self, depth: int = 4, branching: int = 2) -> str:
        """Genera representación de árbol de recursión"""
        recursive_funcs = self.builder.get_recursive_functions()
        
        if not recursive_funcs:
            return "No se detectaron funciones recursivas"
        
        tree_str = "ARBOLES DE RECURSION\n"
        tree_str += "=" * 70 + "\n\n"
        
        for func_name in recursive_funcs:
            tree_str += f"Funcion: {func_name}\n"
            tree_str += self._build_level(depth, "T(n)", branching, 0)
            tree_str += "\n" + "-" * 70 + "\n"
        
        return tree_str
    
    def _build_level(self, depth: int, label: str, branching: int, indent: int) -> str:
        """Construye nivel del árbol"""
        if depth == 0:
            return "  " * indent + "O(1)\n"
        
        result = "  " * indent + f"├─ {label}\n"
        
        for _ in range(branching):
            result += self._build_level(depth - 1, f"T(n/{branching})", branching, indent + 1)
        
        return result


class JsonExporter:
    """Exporta AST a JSON"""
    
    def __init__(self, builder: ASTDiagramBuilder) -> None:
        self.builder = builder
    
    def export(self, filepath: str) -> None:
        """Exporta a JSON"""
        nodes = self.builder.get_nodes()
        data: Dict[str, Any] = {}
        
        for node_id, node in nodes.items():
            data[node_id] = {
                'type': node.node_type,
                'label': node.label,
                'value': node.value,
                'children': node.children,
                'complexity': node.complexity,
                'is_recursive': node.is_recursive,
                'line_number': node.line_number,
                'exec_count': node.exec_count,
                'time_cost': node.time_cost,
                'space_cost': node.space_cost,
            }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"JSON exportado: {filepath}")


def analyze_and_visualize(code: str, code_type: str = "python", output_dir: str = "visualization/output") -> Dict[str, Any]:
    """
    Analiza código y genera diagramas en múltiples formatos.

    Args:
        code (str): Código fuente o pseudocódigo a analizar.
        code_type (str): Tipo de código ("python" o "pseudocode").
        output_dir (str): Directorio donde se guardarán los diagramas generados.

    Returns:
        Dict[str, Any]: Resultados del análisis, incluyendo:
            - code_type (str): Tipo de código analizado.
            - total_nodes (int): Número total de nodos generados.
            - recursive_functions (List[str]): Funciones recursivas detectadas.
            - files (List[str]): Archivos generados (DOT, SVG, PNG, JSON, etc.).
            - complexity (Optional[Dict[str, Any]]): Resultados del análisis de complejidad.
    """
    print(f"Analizando codigo ({code_type})...\n")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    builder = ASTDiagramBuilder()
    
    results: Dict[str, Any] = {
        "code_type": code_type,
        "total_nodes": 0,
        "recursive_functions": [],
        "files": [],
        "complexity": None
    }
    
    # Procesar según tipo
    if code_type == "pseudocode":
        analysis = builder.build_from_pseudocode(code)
        results["complexity"] = analysis.get("complexity")
    else:
        root_id = builder.build_from_python_code(code)
        if not root_id:
            return {"error": "No se pudo parsear el codigo"}
    
    if not builder.get_nodes():
        return {"error": "No se generaron nodos"}
    
    results["total_nodes"] = len(builder.get_nodes())
    results["recursive_functions"] = list(builder.get_recursive_functions())
    
    # Graphviz
    print("Generando Graphviz...")
    gv = GraphvizExporter(builder)
    gv.generate()
    dot_path = f"{output_dir}/ast_diagram.dot"
    gv.save(dot_path)
    results["files"].append(dot_path)
    
    svg_path = f"{output_dir}/ast_diagram.svg"
    if gv.export_svg(dot_path, svg_path):
        results["files"].append(svg_path)
    
    # Mermaid
    print("Generando Mermaid...")
    mm = MermaidExporter(builder)
    mm.generate()
    md_path = f"{output_dir}/ast_diagram.md"
    mm.save_markdown(md_path)
    results["files"].append(md_path)
    
    # NetworkX
    print("Generando PNG...")
    nx_exp = NetworkxExporter(builder)
    png_path = f"{output_dir}/ast_diagram.png"
    if nx_exp.save_png(png_path):
        results["files"].append(png_path)
    
    # Recursion
    print("Analizando recursion...")
    rec_viz = RecursionTreeVisualizer(builder)
    recursion_tree = rec_viz.generate_tree(depth=4, branching=2)
    rec_path = f"{output_dir}/recursion_analysis.txt"
    with open(rec_path, 'w', encoding='utf-8') as f:
        f.write(recursion_tree)
    results["files"].append(rec_path)
    
    # JSON
    print("Exportando JSON...")
    json_exp = JsonExporter(builder)
    json_path = f"{output_dir}/ast_data.json"
    json_exp.export(json_path)
    results["files"].append(json_path)
    
    print(f"\nAnalisis completado en '{output_dir}'\n")
    
    return results


if __name__ == "__main__":
    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    results = analyze_and_visualize(sample_code, code_type="python")
    print(json.dumps(results, indent=2, ensure_ascii=False))