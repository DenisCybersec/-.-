#!/usr/bin/env python3
"""
C Code to Algorithm Flowchart Converter - v2.0.5
CORRECT STRUCTURE: Decompose loops into separate parts with loop_limit nodes

Author: AI Assistant
Version: 2.0.5 (Correct loop decomposition)
Requirements: pycparser, antlr4-python3-runtime
"""

import os
import sys
import re
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from xml.etree import ElementTree as ET
from xml.dom import minidom

import pycparser
from pycparser import c_ast, c_parser


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# GEOMETRY CONFIGURATION
# ============================================================================

@dataclass
class GeometryConfig:
    """Layout configuration"""
    PAGE_WIDTH: int = 827
    PAGE_HEIGHT: int = 1169
    GRID_SIZE: int = 10
    
    # Element dimensions
    START_END_WIDTH: int = 100
    START_END_HEIGHT: int = 60
    PROCESS_WIDTH: int = 180
    PROCESS_HEIGHT: int = 80
    DECISION_WIDTH: int = 140
    DECISION_HEIGHT: int = 100
    LOOP_WIDTH: int = 160
    LOOP_HEIGHT: int = 80
    IO_WIDTH: int = 120
    IO_HEIGHT: int = 60
    DOCUMENT_WIDTH: int = 140
    DOCUMENT_HEIGHT: int = 80
    
    # Spacing
    VERTICAL_SPACING: int = 100
    HORIZONTAL_SPACING: int = 200
    
    # Base position
    BASE_X: int = 100
    BASE_Y: int = 50


# ============================================================================
# FLOWCHART ELEMENT DEFINITIONS
# ============================================================================

@dataclass
class FlowchartNode:
    """Represents a flowchart node"""
    node_id: str
    node_type: str  # 'start_end', 'process', 'decision', 'loop', 'io', 'document'
    label: str
    x: int = 0
    y: int = 0
    width: int = 100
    height: int = 60
    style: str = ""
    parent_id: str = "1"
    
    def get_style(self) -> str:
        """Get mxgraph style string for this node type"""
        styles = {
            "start_end": "strokeWidth=2;html=1;shape=mxgraph.flowchart.start_1;whiteSpace=wrap;",
            "process": "shape=process;whiteSpace=wrap;html=1;backgroundOutline=1;strokeWidth=2;",
            "decision": "rhombus;whiteSpace=wrap;html=1;strokeWidth=2;",
            "loop": "strokeWidth=2;html=1;shape=mxgraph.flowchart.loop_limit;whiteSpace=wrap;",
            "io": "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;strokeWidth=2;",
            "document": "shape=document;whiteSpace=wrap;html=1;boundedLbl=1;strokeWidth=2;",
        }
        return styles.get(self.node_type, "whiteSpace=wrap;html=1;")


@dataclass
class FlowchartEdge:
    """Represents a connection between nodes"""
    edge_id: str
    source_id: str
    target_id: str
    label: str = ""
    curved: bool = False
    
    def get_style(self) -> str:
        """Get style string for edge"""
        return "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"


@dataclass
class FunctionDiagram:
    """Complete diagram for one function"""
    name: str
    func_id: str
    params: List[str] = field(default_factory=list)
    nodes: List[FlowchartNode] = field(default_factory=list)
    edges: List[FlowchartEdge] = field(default_factory=list)
    entry_node_id: str = ""
    exit_node_id: str = ""


# ============================================================================
# CODE STRUCTURE ANALYZER
# ============================================================================

class LoopDecomposer:
    """Decomposes for/while loops into separate parts"""
    
    @staticmethod
    def decompose_for_loop(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Decompose: for(init; condition; increment)
        Returns: (init, condition, increment)
        """
        match = re.search(r'for\s*\((.*?);(.*?);(.*?)\)', line)
        if match:
            init = match.group(1).strip()
            condition = match.group(2).strip()
            increment = match.group(3).strip()
            return (init, condition, increment)
        return (None, None, None)
    
    @staticmethod
    def decompose_while_loop(line: str) -> Optional[str]:
        """
        Extract: while(condition)
        Returns: condition
        """
        match = re.search(r'while\s*\((.*?)\)', line)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def find_loop_body_end(lines: List[str], start_line: int) -> int:
        """Find the line where loop body ends (matching brace)"""
        if '{' not in lines[start_line]:
            return start_line
        
        brace_count = 0
        for i in range(start_line, len(lines)):
            brace_count += lines[i].count('{') - lines[i].count('}')
            if brace_count == 0:
                return i
        
        return len(lines) - 1


# ============================================================================
# ULTRA-ROBUST CODE PREPROCESSING
# ============================================================================

class UltraRobustPreprocessor:
    """Advanced preprocessing for tricky C code"""
    
    @staticmethod
    def clean_code(code: str) -> str:
        """Remove preprocessor directives, comments, and forward declarations"""
        lines = code.split('\n')
        cleaned = []
        in_comment = False
        
        for line in lines:
            if '/*' in line:
                in_comment = True
            if '*/' in line:
                in_comment = False
                continue
            if in_comment:
                continue
            
            if line.strip().startswith('#'):
                continue
            
            if '//' in line:
                line = line[:line.index('//')]
            
            stripped = line.strip()
            
            if stripped.endswith(';') and not stripped.startswith('}'):
                continue
            
            if line.strip():
                cleaned.append(line)
        
        return '\n'.join(cleaned)


# ============================================================================
# CODE EXTRACTION & ANALYSIS
# ============================================================================

class CCodeExtractor:
    """Extracts functions and code from C AST"""
    
    def __init__(self):
        self.functions: Dict[str, c_ast.FuncDef] = {}
        self.function_bodies: Dict[str, List[str]] = {}
        self.function_params: Dict[str, List[str]] = {}
    
    def parse_file(self, filepath: str) -> Optional[c_ast.FileAST]:
        """Parse C file with robust preprocessing"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.parse_code(code, filepath)
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None
    
    def parse_code(self, code: str, filename: str = "input.c") -> Optional[c_ast.FileAST]:
        """Parse C code with preprocessing"""
        parser = c_parser.CParser()
        
        logger.info("Attempting direct parsing...")
        try:
            ast = parser.parse(code, filename=filename)
            self._extract_functions(ast, code)
            logger.info("✓ Parsing successful")
            return ast
        except:
            pass
        
        logger.info("Attempting preprocessing...")
        try:
            cleaned = UltraRobustPreprocessor.clean_code(code)
            ast = parser.parse(cleaned, filename=filename)
            self._extract_functions(ast, cleaned)
            logger.info("✓ Preprocessing successful")
            return ast
        except Exception as e:
            logger.error(f"✗ Parsing failed: {e}")
            self._extract_functions_raw(code)
            return None
    
    def _extract_functions(self, ast: c_ast.FileAST, code: str) -> None:
        """Extract all functions from AST"""
        if not ast or not ast.ext:
            return
        
        self._extract_raw_bodies(code)
        
        for node in ast.ext:
            if isinstance(node, c_ast.FuncDef):
                func_name = node.decl.name
                self.functions[func_name] = node
                
                params = []
                if node.decl.type.args and node.decl.type.args.params:
                    for param in node.decl.type.args.params:
                        if hasattr(param, 'name') and param.name:
                            params.append(param.name)
                        elif hasattr(param, 'type') and hasattr(param.type, 'declname') and param.type.declname:
                            params.append(param.type.declname)
                
                self.function_params[func_name] = params
                logger.info(f"✓ Found function: {func_name}")
    
    def _extract_functions_raw(self, code: str) -> None:
        """Extract functions directly from raw code"""
        logger.info("Extracting functions from source...")
        
        pattern = r'(\w+\s+)+(\w+)\s*\(([^)]*)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        
        for match in re.finditer(pattern, code, re.DOTALL):
            func_name = match.group(2)
            params_str = match.group(3)
            body = match.group(4)
            
            params = []
            if params_str.strip():
                for param in params_str.split(','):
                    param_clean = re.sub(r'(const|volatile|static|extern|restrict)\s+', '', param.strip())
                    words = param_clean.split()
                    if words:
                        var_name = words[-1].replace('*', '').replace('&', '')
                        if var_name:
                            params.append(var_name)
            
            self.function_params[func_name] = params
            
            lines = []
            for line in body.split('\n'):
                line = line.strip()
                if line and not line.startswith('//'):
                    lines.append(line)
            
            self.function_bodies[func_name] = lines
            logger.info(f"✓ Found function: {func_name}")
    
    def _extract_raw_bodies(self, code: str) -> None:
        """Extract raw function bodies from source code"""
        pattern = r'(\w+)\s*\([^)]*\)\s*\{([^}]*)\}'
        
        for match in re.finditer(pattern, code, re.DOTALL):
            func_name = match.group(1)
            body = match.group(2)
            
            lines = []
            for line in body.split('\n'):
                line = line.strip()
                if line and not line.startswith('//'):
                    lines.append(line)
            
            self.function_bodies[func_name] = lines
    
    def get_function_body_lines(self, func_name: str) -> List[str]:
        """Get function body as list of code lines"""
        return self.function_bodies.get(func_name, [])
    
    def get_all_functions(self) -> List[str]:
        """Get list of all function names"""
        return list(self.function_params.keys())


# ============================================================================
# CORRECT FLOWCHART GENERATOR WITH LOOP DECOMPOSITION
# ============================================================================

class CorrectFlowchartBuilder:
    """Builds flowcharts with CORRECT loop decomposition"""
    
    def __init__(self, config: GeometryConfig = None):
        self.config = config or GeometryConfig()
        self.node_counter = 0
        self.edge_counter = 0
    
    def reset_counters(self):
        """Reset counters for new diagram"""
        self.node_counter = 0
        self.edge_counter = 0
    
    def build_function_diagram(self, func_name: str, extractor: CCodeExtractor) -> FunctionDiagram:
        """Build flowchart with CORRECT loop structure"""
        self.reset_counters()
        
        params = extractor.function_params.get(func_name, [])
        params = [p for p in params if p]
        
        diagram = FunctionDiagram(
            name=func_name,
            func_id=str(uuid.uuid4()),
            params=params
        )
        
        # Start node
        if params:
            params_str = ", ".join(params)
            start_label = f"{func_name}({params_str})"
        else:
            start_label = func_name
        
        start_id = self._add_node(diagram, "start_end", start_label, 
                                  self.config.BASE_X, self.config.BASE_Y,
                                  self.config.START_END_WIDTH, self.config.START_END_HEIGHT)
        diagram.entry_node_id = start_id
        current_y = self.config.BASE_Y + self.config.START_END_HEIGHT + self.config.VERTICAL_SPACING
        current_node_id = start_id
        
        # Get function body
        body_lines = extractor.get_function_body_lines(func_name)
        
        # Process each line
        i = 0
        while i < len(body_lines):
            line = body_lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # FOR LOOPS: Decompose into parts!
            if line.startswith('for'):
                init, condition, increment = LoopDecomposer.decompose_for_loop(line)
                
                if init and condition and increment:
                    # Part 1: Initialization
                    loop_id1 = self._add_node(diagram, "loop", init,
                                             self.config.BASE_X, current_y,
                                             self.config.LOOP_WIDTH, self.config.LOOP_HEIGHT)
                    self._add_edge(diagram, current_node_id, loop_id1)
                    current_y += self.config.LOOP_HEIGHT + self.config.VERTICAL_SPACING
                    
                    # Part 2: Condition
                    loop_id2 = self._add_node(diagram, "loop", condition,
                                             self.config.BASE_X, current_y,
                                             self.config.LOOP_WIDTH, self.config.LOOP_HEIGHT)
                    self._add_edge(diagram, loop_id1, loop_id2)
                    current_y += self.config.LOOP_HEIGHT + self.config.VERTICAL_SPACING
                    
                    current_node_id = loop_id2
                    
                    # Find loop body end
                    loop_end = LoopDecomposer.find_loop_body_end(body_lines, i)
                    
                    # Process loop body
                    i += 1
                    while i < loop_end:
                        inner_line = body_lines[i].strip()
                        
                        if inner_line:
                            # Process statement
                            new_id = self._add_node(diagram, "process", inner_line,
                                                   self.config.BASE_X, current_y,
                                                   self.config.PROCESS_WIDTH, self.config.PROCESS_HEIGHT)
                            self._add_edge(diagram, current_node_id, new_id)
                            current_node_id = new_id
                            current_y += self.config.PROCESS_HEIGHT + self.config.VERTICAL_SPACING
                        
                        i += 1
                    
                    # Part 3: Increment
                    loop_id3 = self._add_node(diagram, "loop", increment,
                                             self.config.BASE_X, current_y,
                                             self.config.LOOP_WIDTH, self.config.LOOP_HEIGHT)
                    self._add_edge(diagram, current_node_id, loop_id3)
                    current_node_id = loop_id3
                    current_y += self.config.LOOP_HEIGHT + self.config.VERTICAL_SPACING
                    
                    i += 1
                else:
                    i += 1
            
            # WHILE LOOPS: Single condition
            elif line.startswith('while'):
                condition = LoopDecomposer.decompose_while_loop(line)
                
                if condition:
                    # Condition node
                    loop_id = self._add_node(diagram, "loop", condition,
                                            self.config.BASE_X, current_y,
                                            self.config.LOOP_WIDTH, self.config.LOOP_HEIGHT)
                    self._add_edge(diagram, current_node_id, loop_id)
                    current_y += self.config.LOOP_HEIGHT + self.config.VERTICAL_SPACING
                    
                    current_node_id = loop_id
                    
                    # Find loop body
                    loop_end = LoopDecomposer.find_loop_body_end(body_lines, i)
                    
                    # Process body
                    i += 1
                    while i < loop_end:
                        inner_line = body_lines[i].strip()
                        
                        if inner_line:
                            new_id = self._add_node(diagram, "process", inner_line,
                                                   self.config.BASE_X, current_y,
                                                   self.config.PROCESS_WIDTH, self.config.PROCESS_HEIGHT)
                            self._add_edge(diagram, current_node_id, new_id)
                            current_node_id = new_id
                            current_y += self.config.PROCESS_HEIGHT + self.config.VERTICAL_SPACING
                        
                        i += 1
                    
                    i += 1
                else:
                    i += 1
            
            # IF STATEMENTS: Diamond decision
            elif line.startswith('if') or line.startswith('switch'):
                condition = self._extract_condition(line)
                
                new_id = self._add_node(diagram, "decision", condition,
                                       self.config.BASE_X, current_y,
                                       self.config.DECISION_WIDTH, self.config.DECISION_HEIGHT)
                self._add_edge(diagram, current_node_id, new_id)
                current_node_id = new_id
                current_y += self.config.DECISION_HEIGHT + self.config.VERTICAL_SPACING
                
                # Skip block
                block_end = self._find_block_end(body_lines, i)
                i = block_end + 1
            
            # PRINT/PRINTF
            elif 'printf' in line or 'cout' in line or 'puts' in line:
                new_id = self._add_node(diagram, "document", "Вывод",
                                       self.config.BASE_X, current_y,
                                       self.config.DOCUMENT_WIDTH, self.config.DOCUMENT_HEIGHT)
                self._add_edge(diagram, current_node_id, new_id)
                current_node_id = new_id
                current_y += self.config.DOCUMENT_HEIGHT + self.config.VERTICAL_SPACING
                i += 1
            
            # SCANF
            elif 'scanf' in line or 'cin' in line:
                var_match = re.search(r'&?(\w+)', line)
                var_name = var_match.group(1) if var_match else "ввод"
                new_id = self._add_node(diagram, "io", var_name,
                                       self.config.BASE_X, current_y,
                                       self.config.IO_WIDTH, self.config.IO_HEIGHT)
                self._add_edge(diagram, current_node_id, new_id)
                current_node_id = new_id
                current_y += self.config.IO_HEIGHT + self.config.VERTICAL_SPACING
                i += 1
            
            # REGULAR STATEMENTS
            else:
                line_clean = line.rstrip(';')
                new_id = self._add_node(diagram, "process", line_clean,
                                       self.config.BASE_X, current_y,
                                       self.config.PROCESS_WIDTH, self.config.PROCESS_HEIGHT)
                self._add_edge(diagram, current_node_id, new_id)
                current_node_id = new_id
                current_y += self.config.PROCESS_HEIGHT + self.config.VERTICAL_SPACING
                i += 1
        
        # End node
        end_id = self._add_node(diagram, "start_end", "конец",
                               self.config.BASE_X, current_y,
                               self.config.START_END_WIDTH, self.config.START_END_HEIGHT)
        diagram.exit_node_id = end_id
        self._add_edge(diagram, current_node_id, end_id)
        
        return diagram
    
    def _find_block_end(self, lines: List[str], start: int) -> int:
        """Find matching closing brace"""
        if '{' not in lines[start]:
            return start
        
        count = 0
        for i in range(start, len(lines)):
            count += lines[i].count('{') - lines[i].count('}')
            if count == 0:
                return i
        
        return len(lines) - 1
    
    def _extract_condition(self, line: str) -> str:
        """Extract condition from if/switch"""
        match = re.search(r'if\s*\((.*?)\)', line)
        if match:
            return match.group(1)
        return "Условие"
    
    def _add_node(self, diagram: FunctionDiagram, node_type: str, label: str,
                  x: int, y: int, width: int = None, height: int = None) -> str:
        """Add node to diagram"""
        self.node_counter += 1
        node_id = f"node-{self.node_counter}"
        
        if width is None:
            width = self.config.PROCESS_WIDTH
        if height is None:
            height = self.config.PROCESS_HEIGHT
        
        safe_label = (label
                     .replace('&', '&amp;')
                     .replace('<', '&lt;')
                     .replace('>', '&gt;')
                     .replace('"', '&quot;'))
        
        node = FlowchartNode(
            node_id=node_id,
            node_type=node_type,
            label=safe_label,
            x=x,
            y=y,
            width=width,
            height=height,
            style=FlowchartNode(node_id, node_type, "").get_style()
        )
        diagram.nodes.append(node)
        return node_id
    
    def _add_edge(self, diagram: FunctionDiagram, source_id: str, target_id: str) -> str:
        """Add edge to diagram"""
        self.edge_counter += 1
        edge_id = f"edge-{self.edge_counter}"
        
        edge = FlowchartEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id
        )
        diagram.edges.append(edge)
        return edge_id


# ============================================================================
# DRAW.IO XML GENERATOR
# ============================================================================

class DrawioXMLGenerator:
    """Generates draw.io XML files"""
    
    def __init__(self, config: GeometryConfig = None):
        self.config = config or GeometryConfig()
    
    def generate_xml(self, diagrams: List[FunctionDiagram]) -> str:
        """Generate complete XML"""
        mxfile = ET.Element('mxfile', attrib={
            'host': 'app.diagrams.net',
            'agent': 'C2Algorithm/2.0.5',
            'version': '29.2.7',
            'pages': str(len(diagrams))
        })
        
        for page_num, diagram in enumerate(diagrams):
            self._add_diagram_page(mxfile, diagram, page_num)
        
        xml_str = minidom.parseString(ET.tostring(mxfile)).toprettyxml(indent="  ")
        lines = xml_str.split('\n')[1:]
        return '\n'.join(line for line in lines if line.strip())
    
    def _add_diagram_page(self, mxfile: ET.Element, diagram: FunctionDiagram, page_num: int) -> None:
        """Add diagram page"""
        diagram_elem = ET.SubElement(mxfile, 'diagram', attrib={
            'name': diagram.name,
            'id': diagram.func_id
        })
        
        graph_model = ET.SubElement(diagram_elem, 'mxGraphModel', attrib={
            'grid': '1',
            'page': str(page_num + 1),
            'gridSize': str(self.config.GRID_SIZE),
            'guides': '1',
            'tooltips': '1',
            'connect': '1',
            'arrows': '1',
            'fold': '1',
            'pageScale': '1',
            'pageWidth': str(self.config.PAGE_WIDTH),
            'pageHeight': str(self.config.PAGE_HEIGHT),
            'math': '0',
            'shadow': '0'
        })
        
        root = ET.SubElement(graph_model, 'root')
        ET.SubElement(root, 'mxCell', attrib={'id': '0'})
        ET.SubElement(root, 'mxCell', attrib={'id': '1', 'parent': '0'})
        
        for node in diagram.nodes:
            self._add_node_element(root, node)
        
        for edge in diagram.edges:
            self._add_edge_element(root, edge)
    
    def _add_node_element(self, root: ET.Element, node: FlowchartNode) -> None:
        """Add node element"""
        cell = ET.SubElement(root, 'mxCell', attrib={
            'id': node.node_id,
            'value': node.label,
            'style': node.style,
            'vertex': '1',
            'parent': node.parent_id
        })
        
        ET.SubElement(cell, 'mxGeometry', attrib={
            'x': str(node.x),
            'y': str(node.y),
            'width': str(node.width),
            'height': str(node.height),
            'as': 'geometry'
        })
    
    def _add_edge_element(self, root: ET.Element, edge: FlowchartEdge) -> None:
        """Add edge element"""
        cell = ET.SubElement(root, 'mxCell', attrib={
            'id': edge.edge_id,
            'style': edge.get_style(),
            'edge': '1',
            'parent': '1',
            'source': edge.source_id,
            'target': edge.target_id
        })
        
        ET.SubElement(cell, 'mxGeometry', attrib={
            'relative': '1',
            'as': 'geometry'
        })
    
    def save_to_file(self, xml_str: str, filepath: str) -> bool:
        """Save XML to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            logger.info(f"✓ Saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False


# ============================================================================
# MAIN CONVERTER CLASS
# ============================================================================

class CToAlgorithmConverter:
    """Main converter - CORRECT loop decomposition"""
    
    def __init__(self, config: GeometryConfig = None):
        self.config = config or GeometryConfig()
        self.extractor = CCodeExtractor()
        self.builder = CorrectFlowchartBuilder(config)
        self.xml_gen = DrawioXMLGenerator(config)
    
    def convert_file(self, c_filepath: str, output_filepath: str = None) -> bool:
        """Convert C file to flowcharts"""
        logger.info(f"Processing {c_filepath}")
        
        ast = self.extractor.parse_file(c_filepath)
        
        functions = self.extractor.get_all_functions()
        if not functions:
            logger.error("✗ No functions found")
            return False
        
        logger.info(f"✓ Found {len(functions)} function(s)")
        
        diagrams: List[FunctionDiagram] = []
        
        for func in functions:
            logger.info(f"Building flowchart for: {func}")
            diagram = self.builder.build_function_diagram(func, self.extractor)
            diagrams.append(diagram)
            logger.info(f"✓ {func}: {len(diagram.nodes)} nodes")
        
        xml_str = self.xml_gen.generate_xml(diagrams)
        
        if not output_filepath:
            output_filepath = Path(c_filepath).stem + ".drawio"
        
        success = self.xml_gen.save_to_file(xml_str, output_filepath)
        
        if success:
            logger.info(f"✓ Generated {len(diagrams)} flowchart(s)")
        
        return success


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python c_to_flowchart_v2.0.5.py <c_file> [output_file]")
        print("\nExample:")
        print("  python c_to_flowchart_v2.0.5.py program.c")
        print("\nOutput: Flowcharts with CORRECT loop decomposition")
        sys.exit(1)
    
    c_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.isfile(c_file):
        logger.error(f"File not found: {c_file}")
        sys.exit(1)
    
    converter = CToAlgorithmConverter()
    success = converter.convert_file(c_file, output_file)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
