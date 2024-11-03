import os
import sys
sys.path.insert(0, '/home/abundancia/projetos/unb-ppgi0119/atividade_01')
import importlib
from simple_bpe_tokenizer import SimpleBPETokenizer
import json
import jsonlines
import random
from pathlib import Path
from tqdm import tqdm

# Função para dividir o corpus em treinamento e teste sem carregar tudo em memória
def split_and_process_corpus(input_file, train_file, test_file, train_ratio=0.2):
    train_count = 0
    test_count = 0

    with jsonlines.open(input_file, "r") as reader, \
         jsonlines.open(train_file, "w") as train_writer, \
         jsonlines.open(test_file, "w") as test_writer:

        for item in tqdm(reader, desc="Processando corpus"):
            try:
                # Verifica se o item é um dicionário válido
                if isinstance(item, dict):
                    # Decide se o item vai para treino ou teste
                    if random.random() < train_ratio:
                        train_writer.write(item)
                        train_count += 1
                    else:
                        test_writer.write(item)
                        test_count += 1
            except jsonlines.InvalidLineError:
                print("Linha inválida ignorada.")
    print(f"\nTotal de arquivos para treinamento: {train_count}")
    print(f"Total de arquivos para teste: {test_count}")
    return train_count, test_count

# Função de treinamento do tokenizador
def train_tokenizer_on_file(tokenizer, train_file, vocab_size=512):
    all_text = []
    with jsonlines.open(train_file, "r") as reader:
        for item in reader:
            if "text" in item:
                all_text.append(item["text"])

    # Concatena o texto de treino e realiza o treinamento
    full_text = " ".join(all_text)
    print(f"Tamanho do texto de treinamento: {len(full_text)} caracteres", flush=True)
    tokenizer.train(full_text, vocab_size)

# Função para tokenizar o conjunto de teste e salvar os dados tokenizados
def tokenize_test_data(tokenizer, test_file, output_file):
    with jsonlines.open(test_file, "r") as reader, \
         jsonlines.open(output_file, "w") as writer:

        for item in tqdm(reader, desc="Tokenizando dados de teste"):
            if "text" in item:
                tokens = tokenizer.encode(item["text"])
                writer.write({"tokens": tokens})

# Função principal para processar e rodar o treinamento e testes
# Função principal para processar e rodar o treinamento e testes
def main():
    input_file = os.path.expanduser("~/projetos/unb-ppgi0119/atividade_01/corpus_completo.jsonl")
    train_file = os.path.expanduser("~/projetos/unb-ppgi0119/atividade_01/train_data.jsonl")
    test_file = os.path.expanduser("~/projetos/unb-ppgi0119/atividade_01/test_data.jsonl")
    output_file = os.path.expanduser("~/projetos/unb-ppgi0119/atividade_01/tokenized_test_data.jsonl")

    # Divide o corpus em treino e teste e captura os contadores
    train_count, test_count = split_and_process_corpus(input_file, train_file, test_file, train_ratio=0.2)

    # Instancia o tokenizador e realiza o treinamento
    tokenizer = SimpleBPETokenizer()  
    train_tokenizer_on_file(tokenizer, train_file, vocab_size=512)

    # Tokeniza os dados de teste e salva no arquivo de saída
    tokenize_test_data(tokenizer, test_file, output_file)

    print("Processamento concluído.")
    print(f"Total de JSONs usados no treinamento: {train_count}")
    print(f"Total de JSONs usados nos testes: {test_count}")

if __name__ == "__main__":
    main()

"""
Executando o script...

(UnB) eprioli@GaiaBouddha:~/projetos/unb-ppgi0119/atividade_01$ python corpus_processor.py
Treinando o vocabulário:   6%|███▊                                                        | 31/492 [00:00<00:00, 63768.23it/s]
Texto codificado: [267, 32, 268, 32, 256, 120, 257]
Texto decodificado: exemplo de texto
Processando corpus: 10000it [00:01, 9108.89it/s]

Total de arquivos para treinamento: 2081
Total de arquivos para teste: 7919
Tamanho do texto de treinamento: 14036521 caracteres
Treinando o vocabulário: 0it [00:00, ?it/s]
Tokenizando dados de teste: 7919it [00:07, 1078.42it/s]
Processamento concluído.
Total de JSONs usados no treinamento: 2081
Total de JSONs usados nos testes: 7919
"""