#!/usr/bin/env python3
"""
Script para actualizar todos los notebooks de ChatBedrock a ChatBedrockConverse
con la nueva sintaxis de parámetros directos.
"""

import json
import os
import re
from pathlib import Path

# Configuración de LLMs con nombres descriptivos
LLM_CONFIG = '''from langchain_aws import ChatBedrockConverse

# 1. CONFIGURACIÓN PARA DEEPSEEK-R1 (Razonamiento Complejo)
llm_deepseek_r1 = ChatBedrockConverse(
    model="us.deepseek.r1-v1:0",
    region_name="us-east-1",
    temperature=0.6,
    max_tokens=8192,
    top_p=0.95,
)

# 2. CONFIGURACIÓN PARA DEEPSEEK-V3
llm_deepseek_v3 = ChatBedrockConverse(
    model="us.deepseek.v3-v1:0",
    region_name="us-east-1",
    temperature=0.7,
    max_tokens=4096,
)

# 3. CONFIGURACIÓN PARA LLAMA 4 SCOUT
llm_scout = ChatBedrockConverse(
    model="us.meta.llama4-scout-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# 4. CONFIGURACIÓN PARA LLAMA 4 MAVERICK
llm_maverick = ChatBedrockConverse(
    model="us.meta.llama4-maverick-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# 5. CONFIGURACIÓN PARA AMAZON NOVA LITE
llm_nova_lite = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# 6. CONFIGURACIÓN PARA AMAZON NOVA MICRO
llm_nova_micro = ChatBedrockConverse(
    model="amazon.nova-micro-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# 7. CONFIGURACIÓN PARA AMAZON NOVA PRO
llm_nova_pro = ChatBedrockConverse(
    model="amazon.nova-pro-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# Seleccionar el LLM activo
llm = llm_scout'''


def update_notebook(notebook_path):
    """Actualiza un notebook individual."""
    print(f"Procesando: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    modified = False
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        # Verificar si la celda contiene ChatBedrock
        if 'ChatBedrock' not in source_text:
            continue
        
        # Buscar celdas que importan y configuran ChatBedrock
        if 'from langchain_aws import ChatBedrock' in source_text:
            # Reemplazar todo el bloque de configuración de LLM
            new_source = LLM_CONFIG
            
            # Convertir a lista de líneas con \n
            new_lines = [line + '\n' for line in new_source.split('\n')]
            # La última línea no debe tener \n al final
            if new_lines:
                new_lines[-1] = new_lines[-1].rstrip('\n')
            
            cell['source'] = new_lines
            modified = True
            print(f"  ✓ Actualizada celda de configuración LLM")
        
        # También actualizar referencias individuales de ChatBedrock a ChatBedrockConverse
        elif 'ChatBedrock(' in source_text and 'ChatBedrockConverse' not in source_text:
            # Reemplazar import
            source_text = source_text.replace(
                'from langchain_aws import ChatBedrock',
                'from langchain_aws import ChatBedrockConverse'
            )
            # Reemplazar llamadas con model_id a model y extraer kwargs
            source_text = re.sub(
                r'ChatBedrock\(',
                'ChatBedrockConverse(',
                source_text
            )
            source_text = source_text.replace('model_id=', 'model=')
            
            # Convertir a lista de líneas
            new_lines = [line + '\n' for line in source_text.split('\n')]
            if new_lines:
                new_lines[-1] = new_lines[-1].rstrip('\n')
            
            cell['source'] = new_lines
            modified = True
            print(f"  ✓ Actualizada celda con ChatBedrock")
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"  ✓ Guardado: {notebook_path}")
        return True
    else:
        print(f"  - No se encontraron cambios necesarios")
        return False


def main():
    """Función principal."""
    base_path = Path('/home/juansebas7ian/langchain-academy')
    
    # Lista de notebooks a actualizar
    notebooks = [
        'module-0/basics.ipynb',
        'module-1/chain.ipynb',
        'module-1/router.ipynb',
        'module-1/agent.ipynb',
        'module-1/agent-memory.ipynb',
        'module-2/trim-filter-messages.ipynb',
        'module-3/breakpoints.ipynb',
        'module-3/edit-state-human-feedback.ipynb',
        'module-3/time-travel.ipynb',
        'module-4/parallelization.ipynb',
        'module-4/research-assistant.ipynb',
    ]
    
    updated_count = 0
    for notebook_rel in notebooks:
        notebook_path = base_path / notebook_rel
        if notebook_path.exists():
            if update_notebook(notebook_path):
                updated_count += 1
        else:
            print(f"⚠ No encontrado: {notebook_path}")
    
    print(f"\n{'='*50}")
    print(f"Total notebooks actualizados: {updated_count}/{len(notebooks)}")
    print("="*50)


if __name__ == '__main__':
    main()
