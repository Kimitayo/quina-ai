import os
from itertools import combinations
from collections import Counter
import statistics

ARQUIVO_DADOS = 'quina.csv'

def minerar_quina():
    print(f"--- ⛏️  MINERAÇÃO DE DADOS AVANÇADA: QUINA (1-80) ---")
    dataset = []
    
    if not os.path.exists(ARQUIVO_DADOS):
        print(f"❌ Erro: Arquivo '{ARQUIVO_DADOS}' não encontrado.")
        return

    try:
        with open(ARQUIVO_DADOS, 'r', encoding='utf-8', errors='ignore') as f:
            linhas = f.readlines()
            for linha in linhas:
                linha = linha.strip()
                if not linha: continue
                
                partes = linha.split(';') # Força separador correto
                nums = []
                
                # Pula as colunas 0 (Concurso) e 1 (Data)
                # Lê apenas da coluna 2 em diante (as bolas)
                for p in partes[2:]:
                    try:
                        n = int(p)
                        if 1 <= n <= 80: nums.append(n)
                    except: continue
                
                if len(nums) >= 5: 
                    dataset.append(sorted(nums)) # Salva ordenado
                    
    except Exception as e:
        print(f"Erro na leitura: {e}")
        return
    
    total_jogos = len(dataset)
    print(f"📊 Total de jogos analisados: {total_jogos}")
    
    if total_jogos == 0:
        print("⚠️ Nenhum jogo válido encontrado.")
        return

    #  1. FREQUÊNCIA DE PARES 
    todos_pares = []
    for jogo in dataset: 
        todos_pares.extend(combinations(jogo, 2))
    freq_pares = Counter(todos_pares)
    
    print("\n🔥 TOP 5 DUQUES (PARES) MAIS FREQUENTES:")
    for par, qtd in freq_pares.most_common(5):
        print(f"   Par {par}: Saiu {qtd} vezes")

    # 2. FREQUÊNCIA DE TERNOS (TRIOS)
    todos_ternos = []
    for jogo in dataset:
        todos_ternos.extend(combinations(jogo, 3))
    freq_ternos = Counter(todos_ternos)

    print("\n🔥 TOP 5 TERNOS (TRIOS) MAIS FREQUENTES:")
    print("   (Dica: Esses trios são a base para jogos fortes)")
    for trio, qtd in freq_ternos.most_common(5):
        print(f"   Trio {trio}: Saiu {qtd} vezes")

    #  3. ANÁLISE DE SOMA 
    somas = [sum(jogo) for jogo in dataset]
    if somas:
        media_soma = statistics.mean(somas)
        min_soma_comum = min(somas)
        max_soma_comum = max(somas)
        
        print("\n🧮 ESTATÍSTICAS DE SOMA (Para calibrar sua IA):")
        print(f"   Média histórica da soma: {media_soma:.1f}")
        print(f"   Soma Mínima registrada: {min_soma_comum}")
        print(f"   Soma Máxima registrada: {max_soma_comum}")

    #  4. OS ATRASADOS 
    atrasos = {n: 0 for n in range(1, 81)}
    encontrados = set()
    # Varre do mais recente para o mais antigo
    for i, jogo in enumerate(dataset[::-1]):
        for n in jogo:
            if n not in encontrados:
                atrasos[n] = i
                encontrados.add(n)
        if len(encontrados) == 80: break
    
    print("\n⏰ TOP 10 MAIS ATRASADOS (Dorminhocos):")
    atrasados_ord = sorted(atrasos.items(), key=lambda x: x[1], reverse=True)
    for n, t in atrasados_ord[:10]:
        print(f"   Dezena {n:02d}: {t} concursos sem sair")

if __name__ == "__main__":
    minerar_quina()