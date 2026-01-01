import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURAÇÕES OTIMIZADAS QUINA ---
ARQUIVO_DADOS = 'quina.csv'
ARQUIVO_LSTM = 'quina_cerebro_lstm.keras'
ARQUIVO_GRU = 'quina_cerebro_gru.keras'

WINDOW_SIZE = 20       # Quina precisa de histórico maior que Loto
NUM_NUMEROS = 80       # Universo 1-80

# --- FEATURES AVANÇADAS ---
# 1.Pares, 2.Soma, 3.Primos, 4.Fibo, 5.Repetidos, 6.Amplitude
# 7 a 14: As 8 faixas de dezenas (01-10, 11-20 ... 71-80)
NUM_FEATURES_EXTRAS = 14
INPUT_DIM = NUM_NUMEROS + NUM_FEATURES_EXTRAS 

# Tabelas Matemáticas (Até 80)
PRIMOS = set([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 
              61, 67, 71, 73, 79])
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55])

def carregar_dados():
    print(f"🔄 Processando arquivo {ARQUIVO_DADOS}...")
    dataset_final = []
    
    if not os.path.exists(ARQUIVO_DADOS):
        print("❌ ERRO: Arquivo 'quina.csv' não encontrado.")
        return []

    try:
        with open(ARQUIVO_DADOS, 'r', encoding='utf-8', errors='ignore') as f:
            linhas = f.readlines()
            for linha in linhas:
                linha = linha.strip()
                if not linha: continue
                
                # --- ALTERAÇÃO AQUI: BLINDAGEM DO CSV ---
                # 1. Forçamos o separador ';'
                partes = linha.split(';')
                
                # 2. Ignoramos as colunas 0 (Concurso) e 1 (Data)
                # O loop começa da parte 2 em diante
                nums = []
                for p in partes[2:]:
                    try:
                        n = int(p)
                        if 1 <= n <= 80: nums.append(n)
                    except: continue
                
                # 3. Se achou 5 números, salva (já ordenado)
                if len(nums) >= 5:
                    dataset_final.append(sorted(nums))
        
        print(f"✅ Sucesso! {len(dataset_final)} concursos carregados.")
        return dataset_final
    except Exception as e:
        print(f"❌ Erro de leitura: {e}")
        return []

def calcular_features_extras(jogos):
    features = []
    max_soma = 390 # 76+77+78+79+80
    
    for i in range(len(jogos)):
        jogo = jogos[i]
        
        qtd_pares = sum(1 for n in jogo if n % 2 == 0)
        soma = sum(jogo)
        qtd_primos = sum(1 for n in jogo if n in PRIMOS)
        qtd_fibo = sum(1 for n in jogo if n in FIBONACCI)
        
        if i > 0:
            repetidos = sum(1 for n in jogo if n in set(jogos[i-1]))
        else:
            repetidos = 0 # Quina raramente repete do anterior, então 0 é safe
        
        amplitude = max(jogo) - min(jogo)

        # --- NOVA LÓGICA: DEZENAS (Historiograma) ---
        # Conta quantos números caíram em cada década (1-10, 11-20...)
        dezenas = [0] * 8
        for n in jogo:
            idx = (n - 1) // 10 # Retorna 0 para 1-10, 1 para 11-20, etc.
            if 0 <= idx < 8:
                dezenas[idx] += 1
        
        # Normalização (divide por 5.0 pois são 5 bolas)
        dados_dezenas = [d / 5.0 for d in dezenas]

        # Monta a linha de features
        linha_feat = [
            qtd_pares/5.0, 
            soma/max_soma, 
            qtd_primos/5.0, 
            qtd_fibo/5.0, 
            repetidos/5.0,
            amplitude/79.0
        ] + dados_dezenas # Adiciona as 8 colunas das dezenas
        
        features.append(linha_feat)
        
    return np.array(features)

def criar_modelo(tipo, input_shape):
    model = Sequential(name=f"Quina_{tipo}")
    if tipo == 'LSTM': Layer = LSTM
    else: Layer = GRU
    
    model.add(Bidirectional(Layer(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(Layer(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Layer(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_NUMEROS, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("🚀 INICIANDO SISTEMA DE TREINAMENTO QUINA 🚀")
    jogos = carregar_dados()
    
    if len(jogos) > WINDOW_SIZE + 50:
        print("📊 Engenharia de Atributos (Mapeando Dezenas)...")
        mlb = MultiLabelBinarizer(classes=range(1, NUM_NUMEROS + 1))
        dados_numeros = mlb.fit_transform(jogos)
        dados_features = calcular_features_extras(jogos)
        dados_completos = np.hstack((dados_numeros, dados_features))

        X, y = [], []
        for i in range(WINDOW_SIZE, len(dados_completos)):
            X.append(dados_completos[i-WINDOW_SIZE:i])
            y.append(dados_numeros[i])
        X, y = np.array(X), np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
        print(f"🧠 Dados prontos. Treinando com {len(X_train)} sequências.")

        print("\n>>> Treinando LSTM...")
        model_lstm = criar_modelo('LSTM', (WINDOW_SIZE, INPUT_DIM))
        ckpt_lstm = ModelCheckpoint(ARQUIVO_LSTM, monitor='loss', save_best_only=True, verbose=0)
        model_lstm.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), callbacks=[ckpt_lstm], verbose=1)
        
        print("\n>>> Treinando GRU...")
        model_gru = criar_modelo('GRU', (WINDOW_SIZE, INPUT_DIM))
        ckpt_gru = ModelCheckpoint(ARQUIVO_GRU, monitor='loss', save_best_only=True, verbose=0)
        model_gru.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), callbacks=[ckpt_gru], verbose=1)

        print("\n✅ TREINAMENTO CONCLUÍDO!")
    else:
        print("❌ Dados insuficientes.")