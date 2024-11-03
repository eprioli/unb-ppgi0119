import json
import os
import jsonlines
from pathlib import Path
from tqdm import tqdm

def merge_json_files(input_dir, output_file):
    # Cria o arquivo de saída JSONL
    with jsonlines.open(output_file, mode='w') as writer:
        # Percorre todos os arquivos JSON no diretório de entrada
        for json_file in tqdm(Path(input_dir).glob("*.json"), desc="Processando arquivos JSON"):
            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)  # Carrega o conteúdo JSON
                    writer.write(data)  # Escreve no arquivo JSONL
                except json.JSONDecodeError:
                    print(f"Erro ao ler o arquivo: {json_file}")

# Caminho do diretório onde os JSONs estão localizados e do arquivo de saída
input_dir = os.path.expanduser("~/projetos/unb-ppgi0119/atividade_01/corpus_raw")
output_file = os.path.expanduser("~/projetos/unb-ppgi0119/atividade_01/corpus_completo.jsonl")

# Executa a função de mesclagem
merge_json_files(input_dir, output_file)

# Executando esse script
"""
(UnB) eprioli@GaiaBouddha:~/projetos/unb-ppgi0119/atividade_01$ python merge_json_files.py
Processando arquivos JSON: 8172it [00:05, 1611.42it/s]
"""