import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
import os
from itertools import combinations

#  CONFIGURAÇÕES IDÊNTICAS AO MAIN 
ARQUIVO_DADOS = 'quina.csv'
ARQUIVO_LSTM = 'quina_cerebro_lstm.keras'
ARQUIVO_GRU = 'quina_cerebro_gru.keras'
WINDOW_SIZE = 20
NUM_NUMEROS = 80
NUM_FEATURES_EXTRAS = 14 # Sincronizado com main.py
INPUT_DIM = NUM_NUMEROS + NUM_FEATURES_EXTRAS

MIN_SOMA = 120
MAX_SOMA = 290
MIN_PARES = 1
MAX_PARES = 4
MIN_AMPLITUDE = 20   

PRIMOS = set([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 
              61, 67, 71, 73, 79])
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55])

def carregar_ultimos_jogos():
    dataset_final = []
    if not os.path.exists(ARQUIVO_DADOS): return []
    try:
        with open(ARQUIVO_DADOS, 'r', encoding='utf-8', errors='ignore') as f:
            linhas = f.readlines()
            for linha in linhas:
                linha = linha.strip()
                if not linha: continue
                
                partes = linha.split(';') # Força ponto e vírgula
                nums = []
                # Pula as colunas 0 (Concurso) e 1 (Data)
                for p in partes[2:]:
                    try:
                        n = int(p)
                        if 1 <= n <= 80: nums.append(n)
                    except: continue
                
                if len(nums) >= 5:
                    dataset_final.append(sorted(nums)) # Salva ordenado
        
        if len(dataset_final) >= WINDOW_SIZE:
            return dataset_final[-WINDOW_SIZE:]
        else: return []
    except: return []

def calcular_features_extras(jogos):
    features = []
    max_soma = 390
    for i in range(len(jogos)):
        jogo = jogos[i]
        qtd_pares = sum(1 for n in jogo if n % 2 == 0)
        soma = sum(jogo)
        qtd_primos = sum(1 for n in jogo if n in PRIMOS)
        qtd_fibo = sum(1 for n in jogo if n in FIBONACCI)
        
        if i > 0:
            repetidos = sum(1 for n in jogo if n in set(jogos[i-1]))
        else: repetidos = 0
        
        amplitude = max(jogo) - min(jogo)

        dezenas = [0] * 8
        for n in jogo:
            idx = (n - 1) // 10
            if 0 <= idx < 8: dezenas[idx] += 1
        dados_dezenas = [d / 5.0 for d in dezenas]

        linha = [
            qtd_pares/5.0, soma/max_soma, qtd_primos/5.0, qtd_fibo/5.0, repetidos/5.0,
            amplitude/79.0
        ] + dados_dezenas
        features.append(linha)
    return np.array(features)

def validar_jogo(jogo):
    soma = sum(jogo)
    pares = sum(1 for n in jogo if n % 2 == 0)
    amplitude = max(jogo) - min(jogo)
    
    if not (MIN_SOMA <= soma <= MAX_SOMA): return False
    if not (MIN_PARES <= pares <= MAX_PARES): return False
    if amplitude < MIN_AMPLITUDE: return False
    
    # Filtro: Evita concentração exagerada em uma única década
    dezenas_count = [0]*8
    for n in jogo: dezenas_count[(n-1)//10] += 1
    if max(dezenas_count) >= 4: return False 

    return True

if __name__ == "__main__":
    print("\n==============================================")
    print("      QUINA PREDICTOR SYSTEM (NÍVEL 5)        ")
    print("==============================================\n")

    try:
        if not os.path.exists(ARQUIVO_LSTM) or not os.path.exists(ARQUIVO_GRU):
            raise FileNotFoundError("⚠️ Modelos não encontrados. Rode main.py!")

        model_lstm = load_model(ARQUIVO_LSTM)
        model_gru = load_model(ARQUIVO_GRU)
        ultimos_jogos = carregar_ultimos_jogos()

        if len(ultimos_jogos) == WINDOW_SIZE:
            
            # Prepara os dados da memória recente
            mlb = MultiLabelBinarizer(classes=range(1, NUM_NUMEROS + 1))
            mlb.fit([list(range(1, 81))])
            dados_numeros = mlb.transform(ultimos_jogos)
            dados_features = calcular_features_extras(ultimos_jogos)
            input_combined = np.hstack((dados_numeros, dados_features))
            input_reshaped = input_combined.reshape(1, WINDOW_SIZE, INPUT_DIM)

            print("🤖 Consultando Especialistas (LSTM + GRU)...")
            pred_lstm = model_lstm.predict(input_reshaped, verbose=0)[0]
            pred_gru = model_gru.predict(input_reshaped, verbose=0)[0]
            predicao_final = (pred_lstm + pred_gru) / 2.0

            # SELEÇÃO DO POOL (30 Dezenas)
            indices_top = predicao_final.argsort()[-30:][::-1]
            dezenas_ouro = sorted([i + 1 for i in indices_top])
            
            print(f"\n💎 Pool Expandido (30 dezenas): {dezenas_ouro}")
            print("⚙️  Processando combinações e aplicando filtros...")
            
            todas_combinacoes = combinations(dezenas_ouro, 5)
            
            jogos_rankeados = []
            cont = 0
            for jogo in todas_combinacoes:
                cont += 1
                if validar_jogo(jogo):
                    score = sum(predicao_final[n-1] for n in jogo)
                    jogos_rankeados.append({'numeros': jogo, 'score': score})
            
            jogos_rankeados.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"   -> Jogos analisados: {cont}")
            print(f"   -> Jogos aprovados: {len(jogos_rankeados)}")

            print("\n" + "="*50)
            print("🏆  TOP 3 PALPITES QUINA  🏆")
            print("="*50)
            
            for i in range(min(3, len(jogos_rankeados))):
                jg = jogos_rankeados[i]
                print(f"#{i+1}: {list(jg['numeros'])}")
                print(f"    (Confiança: {jg['score']:.2f})")
                print("-" * 50)

            print("\n📊 ZONA QUENTE (TOP 10)")
            lista_prob = [(i+1, predicao_final[i]*100) for i in range(NUM_NUMEROS)]
            lista_prob.sort(key=lambda x: x[1], reverse=True)
            for i in range(10):
                print(f"   Dezena {lista_prob[i][0]:02d}: {lista_prob[i][1]:.2f}%")

        else:
            print(f"⚠️ Erro: CSV insuficiente. Precisa de {WINDOW_SIZE} jogos, encontrei {len(ultimos_jogos)}.")

    except Exception as e:
        print(f"❌ Erro: {e}")