#!/usr/bin/env python3
"""
Script para corregir notebooks donde las conversaciones empiezan con AIMessage.
Bedrock Converse API requiere que la conversación empiece con HumanMessage.
"""

import json
import re
from pathlib import Path


def fix_messages_in_notebook(notebook_path):
    """Corrige celdas donde los mensajes empiezan con AIMessage."""
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
        
        original_text = source_text
        
        # Patron 1: messages = [AIMessage(...)]
        # Corregir: messages = [AIMessage(f"So you said you were researching ocean mammals?", name="Bot")]
        if 'messages = [AIMessage(f"So you said you were researching ocean mammals?"' in source_text or \
           'messages = [AIMessage(content=f"So you said you were researching ocean mammals?"' in source_text:
            source_text = source_text.replace(
                'messages = [AIMessage(f"So you said you were researching ocean mammals?", name="Bot")]',
                'messages = [HumanMessage(f"I\'m researching ocean mammals.", name="Lance")]\nmessages.append(AIMessage(f"So you said you were researching ocean mammals?", name="Bot"))'
            )
            source_text = source_text.replace(
                'messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]',
                'messages = [HumanMessage(content=f"I\'m researching ocean mammals.", name="Lance")]\nmessages.append(AIMessage(content=f"So you said you were researching ocean mammals?", name="Model"))'
            )
        
        # Patron 2: messages = [AIMessage("Hi.", name="Bot", id="1")]
        if 'messages = [AIMessage("Hi.", name="Bot", id="1")]' in source_text:
            source_text = source_text.replace(
                'messages = [AIMessage("Hi.", name="Bot", id="1")]',
                'messages = [HumanMessage("Hello!", name="Lance", id="0")]\nmessages.append(AIMessage("Hi.", name="Bot", id="1"))'
            )
        
        # Patron 3: initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
        if 'initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),' in source_text:
            source_text = source_text.replace(
                'initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),',
                'initial_messages = [HumanMessage(content="Hello!", name="Lance"),\n                     AIMessage(content="Hello! How can I assist you?", name="Model"),'
            )
        
        # Patron 4: initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
        if 'initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),' in source_text:
            source_text = source_text.replace(
                'initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),',
                'initial_messages = [HumanMessage(content="Hello!", name="Lance", id="0"),\n                     AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),'
            )
        
        if source_text != original_text:
            # Convertir a lista de líneas
            new_lines = [line + '\n' for line in source_text.split('\n')]
            if new_lines:
                new_lines[-1] = new_lines[-1].rstrip('\n')
            cell['source'] = new_lines
            modified = True
            print(f"  ✓ Corregida celda con mensajes")
    
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
    
    # Lista de notebooks a revisar
    notebooks = [
        'module-1/chain.ipynb',
        'module-2/trim-filter-messages.ipynb',
        'module-2/state-reducers.ipynb',
    ]
    
    updated_count = 0
    for notebook_rel in notebooks:
        notebook_path = base_path / notebook_rel
        if notebook_path.exists():
            if fix_messages_in_notebook(notebook_path):
                updated_count += 1
        else:
            print(f"⚠ No encontrado: {notebook_path}")
    
    print(f"\n{'='*50}")
    print(f"Total notebooks corregidos: {updated_count}/{len(notebooks)}")
    print("="*50)


if __name__ == '__main__':
    main()
