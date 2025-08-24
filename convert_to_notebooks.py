#!/usr/bin/env python3
"""
Convert Python scripts to Jupyter notebooks
Automatically converts .py files in project/ folder to .ipynb format
"""

import json
import os
import re
from pathlib import Path

def python_to_notebook(py_file_path, nb_file_path):
    """Convert a Python file to Jupyter notebook format"""
    
    with open(py_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into cells
    cells = []
    current_cell = []
    current_cell_type = 'code'
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for docstring at the beginning (convert to markdown)
        if i == 0 and line.strip().startswith('"""'):
            # Multi-line docstring at start - convert to markdown
            markdown_lines = []
            i += 1  # Skip opening """
            
            while i < len(lines) and not lines[i].strip().endswith('"""'):
                markdown_lines.append(lines[i])
                i += 1
            
            if i < len(lines):
                i += 1  # Skip closing """
            
            # Create markdown cell
            if markdown_lines:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": markdown_lines
                })
            continue
        
        # Check for section comments (convert to markdown)
        if line.strip().startswith('# ') and len(line.strip()) > 2:
            # Save current code cell if it has content
            if current_cell and any(l.strip() for l in current_cell):
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": current_cell
                })
                current_cell = []
            
            # Create markdown cell for section header
            header_text = line.strip()[2:].strip()
            if not header_text.startswith('#'):
                header_text = f"## {header_text}"
            
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [header_text]
            })
            i += 1
            continue
        
        # Check for function definitions (start new code cell)
        if line.strip().startswith('def ') or line.strip().startswith('class '):
            # Save current cell if it has content
            if current_cell and any(l.strip() for l in current_cell):
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": current_cell
                })
                current_cell = []
        
        # Add line to current cell
        current_cell.append(line)
        i += 1
    
    # Add final cell if it has content
    if current_cell and any(l.strip() for l in current_cell):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": current_cell
        })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write notebook file
    with open(nb_file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {py_file_path} → {nb_file_path}")

def convert_project_files():
    """Convert all Python files in project/ folder to notebooks"""
    project_dir = Path("project")
    
    if not project_dir.exists():
        print("❌ Project directory not found")
        return
    
    py_files = list(project_dir.glob("*.py"))
    
    if not py_files:
        print("❌ No Python files found in project/ directory")
        return
    
    print(f"🔄 Converting {len(py_files)} Python files to notebooks...")
    
    for py_file in py_files:
        # Create notebook filename
        nb_file = py_file.with_suffix('.ipynb')
        
        try:
            python_to_notebook(py_file, nb_file)
        except Exception as e:
            print(f"❌ Error converting {py_file}: {str(e)}")
    
    print(f"\n✅ Conversion complete! Created {len(py_files)} notebooks in project/ folder")

if __name__ == "__main__":
    convert_project_files()
