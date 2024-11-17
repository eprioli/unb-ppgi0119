import os
import math
import nltk
from sklearn.model_selection import train_test_split
import jsonlines
from collections import defaultdict, Counter
import random

# Baixe os recursos necessários do NLTK (execute apenas uma vez)
nltk.download('punkt')

# Caminho para o arquivo de dados
data_file = os.path.expanduser("~/projetos/unb-ppgi0119/atividade_01/train_data.jsonl")

# Função para carregar e tokenizar o texto do arquivo
def load_and_tokenize_data(file_path):
    all_text = []
    
    # Lê o arquivo JSONL e extrai o texto
    with jsonlines.open(file_path, "r") as reader:
        all_text = [item["text"] for item in reader if "text" in item]
    
    # Concatena todo o texto e realiza a tokenização
    full_text = " ".join(all_text)
    return nltk.word_tokenize(full_text)

# Carregar e tokenizar os dados
tokens = load_and_tokenize_data(data_file)

# Dividir os tokens em treino (80%) e teste (20%)
train_tokens, test_tokens = train_test_split(tokens, test_size=0.2, random_state=42)

# Função para construir o modelo de bigrama
def build_bigram_model(tokens):
    model = defaultdict(Counter)
    
    # Conta a frequência dos bigramas
    for w1, w2 in nltk.bigrams(tokens):
        model[w1][w2] += 1

    # Converte contagens em probabilidades
    for w1 in model:
        total_count = sum(model[w1].values())
        model[w1] = {w2: count / total_count for w2, count in model[w1].items()}
    
    return model

# Construir o modelo de bigrama
bigram_model = build_bigram_model(train_tokens)

# Função para calcular a perplexidade do modelo
def calculate_perplexity(model, tokens):
    log_prob_sum = 0
    N = len(tokens)
    
    # Calcula a soma dos logaritmos das probabilidades
    for w1, w2 in nltk.bigrams(tokens):
        prob = model[w1].get(w2, 1e-6)  # Usa uma probabilidade muito pequena se a palavra não for encontrada
        log_prob_sum += -math.log(prob)
    
    # Retorna a perplexidade
    return math.exp(log_prob_sum / N)

# Calcular e exibir a perplexidade
perplexity = calculate_perplexity(bigram_model, test_tokens)
print(f"Perplexidade: {perplexity:.2f}")

# Função para gerar texto a partir do modelo de bigrama
def generate_text(model, start_word, num_words=20):
    current_word = start_word
    text = [current_word]
    
    # Gera palavras com base no modelo de bigrama
    for _ in range(num_words - 1):
        next_words = list(model[current_word].keys())
        probabilities = list(model[current_word].values())
        
        # Se não houver próximas palavras, encerra a geração
        if not next_words:
            break
        
        # Seleciona a próxima palavra com base nas probabilidades
        current_word = random.choices(next_words, probabilities)[0]
        text.append(current_word)
    
    return " ".join(text)

# Gerar e exibir um exemplo de texto com 20 palavras
start_word = random.choice(train_tokens)
generated_text = generate_text(bigram_model, start_word)
print("\nTexto gerado:")
print(generated_text)

