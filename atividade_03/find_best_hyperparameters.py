import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import os
import time
import psutil

# Função para carregar os dados
def load_data(file_paths):
    data_frames = []
    for file_path in file_paths:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data_frames.append(df)
        else:
            raise ValueError("Formato de arquivo não suportado. Use .csv.")
    return pd.concat(data_frames, ignore_index=True)

# Caminhos dos arquivos
data_files = [
    "/home/abundancia/projetos/unb-ppgi0119/atividade_03/re8.csv",
    "/home/abundancia/projetos/unb-ppgi0119/atividade_03/Industry_Sector.csv"
]
output_dir = "/home/abundancia/projetos/unb-ppgi0119/atividade_03/output"  # Diretório de saída
os.makedirs(output_dir, exist_ok=True)

# Função para executar o Greedy Search
def perform_greedy_search(model, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(model, param_grid, scoring='f1_macro', cv=cv, verbose=1, n_jobs=2)  # Limite de n_jobs
    grid_search.fit(X_train, y_train)
    return grid_search

# Configuração dos modelos e seus hiperparâmetros
models = {
    "MultinomialNB": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.1, 0.5]
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "params": {
            "C": [0.1, 1],
            "penalty": ["l2"]
        }
    }
}

# Adicionar cronômetro
start_time = time.time()

# Criar um arquivo para salvar informações de desempenho
performance_log = os.path.join(output_dir, "performance_log.csv")
with open(performance_log, "w") as log_file:
    log_file.write("Step,Elapsed Time (s),CPU Usage (%),Memory Usage (%)\n")

# Função para registrar o desempenho
def log_performance(step):
    elapsed_time = time.time() - start_time
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    with open(performance_log, "a") as log_file:
        log_file.write(f"{step},{elapsed_time:.2f},{cpu_usage},{memory_usage}\n")
    print(f"[LOG] {step}: {elapsed_time:.2f}s, CPU: {cpu_usage}%, Memory: {memory_usage}%")

# Carregar a base de dados
print(f"Processando arquivos: {data_files}")
data = load_data(data_files)
log_performance("Data Loading")

# Ajustar colunas e remover 'file_name'
data = data.rename(columns={"text": "text", "class": "label"})
data = data.drop(columns=["file_name"], errors='ignore')

# Remover classes com menos de 2 amostras
print("Removendo classes com menos de 2 amostras...")
class_counts = data['label'].value_counts()
valid_classes = class_counts[class_counts > 1].index
data = data[data['label'].isin(valid_classes)]
print(f"Classes válidas restantes: {len(valid_classes)}")
log_performance("Class Filtering")

# Divisão da base em treino e teste
print("Dividindo os dados em treino e teste...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
print(f"Tamanho do conjunto de treino: {len(train_data)}, teste: {len(test_data)}")

# Salvar conjuntos de treino e teste
train_data.to_csv(os.path.join(output_dir, "combined_train_data.csv"), index=False)
test_data.to_csv(os.path.join(output_dir, "combined_test_data.csv"), index=False)
print("Dados de treino e teste salvos com sucesso.")
log_performance("Data Splitting")

# Preparar os dados para modelagem
print("Vetorizando os dados...")
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']
print("Vetorização concluída.")
log_performance("Vectorization")

# DataFrame para armazenar os resultados
results = []

# Executar o Greedy Search para cada modelo
for model_name, model_info in models.items():
    print(f"Iniciando Greedy Search para: {model_name} nos arquivos combinados")
    start_model_time = time.time()
    search = perform_greedy_search(model_info['model'], model_info['params'], X_train, y_train)
    elapsed_model_time = time.time() - start_model_time
    print(f"Greedy Search para {model_name} concluído em {elapsed_model_time:.2f} segundos.")
    log_performance(f"Greedy Search - {model_name}")
    for params, mean_score in zip(search.cv_results_['params'], search.cv_results_['mean_test_score']):
        results.append({
            "model": model_name,
            "params": params,
            "f1_macro": mean_score
        })

# Criar um DataFrame com os resultados
print("Salvando resultados do Greedy Search...")
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "combined_greedy_search_results.csv"), index=False)
log_performance("Results Saving")

# Finalizar cronômetro
total_time = time.time() - start_time
print(f"Processo concluído em {total_time:.2f} segundos. Resultados salvos.")
log_performance("Script Completion")
