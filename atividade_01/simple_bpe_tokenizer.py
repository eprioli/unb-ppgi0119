import json
from collections import Counter
from tqdm import tqdm

class SimpleBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    def get_stats(self, tokens):
        # Conta a frequência de cada par de tokens adjacentes
        pairs = Counter()
        for word in tokens:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def merge_vocab(self, pair, tokens):
        # Realiza a fusão de um par específico de tokens
        new_tokens = []
        bigram = ' '.join(pair)
        for word in tokens:
            new_word = ' '.join(word)
            new_word = new_word.replace(bigram, ''.join(pair))
            new_tokens.append(new_word.split())
        return new_tokens

    def train(self, text, vocab_size):
        # Inicialização do vocabulário com tokens do texto
        tokens = [[ch for ch in word] for word in text.split()]
        vocab = Counter(text)
    
        # Adiciona o token `[UNK]` ao vocabulário
        self.vocab["[UNK]"] = len(self.vocab)

        # Continuação do treinamento do vocabulário
        for _ in tqdm(range(vocab_size - len(set(text))), desc="Treinando o vocabulário"):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            tokens = self.merge_vocab(best, tokens)
            self.merges[best] = len(self.merges) + 256

        # Finaliza a criação do vocabulário
        for token in set(text):
            self.vocab[token] = ord(token)
        for i, merge in enumerate(self.merges):
            self.vocab[''.join(merge)] = 256 + i


    def encode(self, text):
        # Codifica o texto em tokens numéricos com base no vocabulário treinado
        tokens = [ch for ch in text]
        for pair in self.merges:
            bigram = ''.join(pair)
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = bigram
                    del tokens[i + 1]
                i += 1
        # Adiciona verificação para tokens ausentes
        return [self.vocab.get(token, self.vocab.get("[UNK]", -1)) for token in tokens]

    def decode(self, ids):
        # Decodifica tokens numéricos de volta para texto
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab[id_] for id_ in ids]
        return ''.join(tokens)

# Exemplo de uso
tokenizer = SimpleBPETokenizer()
sample_text = "Este é um exemplo de texto para treinar o tokenizador."
tokenizer.train(sample_text, vocab_size=512)

# Codificar e decodificar um texto
encoded_text = tokenizer.encode("exemplo de texto")
print("Texto codificado:", encoded_text)

decoded_text = tokenizer.decode(encoded_text)
print("Texto decodificado:", decoded_text)
