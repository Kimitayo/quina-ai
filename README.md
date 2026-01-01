# 🔮 Quina AI Predictor (Deep Learning)

> Um sistema de Inteligência Artificial avançado projetado para analisar padrões não-lineares e séries temporais na loteria **Quina** (Caixa Econômica Federal).

## 🧠 Sobre o Projeto

A Quina possui um universo de **80 números**, o que torna a previsão extremamente complexa devido à alta entropia. Este projeto não busca "adivinhar" o sorteio, mas sim **reduzir drasticamente o espaço de busca** utilizando Redes Neurais Recorrentes.

O sistema opera com uma arquitetura híbrida:
1.  **LSTM (Long Short-Term Memory):** Para aprender tendências de longo prazo.
2.  **GRU (Gated Recurrent Unit):** Para capturar a volatilidade recente.
3.  **Filtros Estatísticos:** Para eliminar combinações matematicamente improváveis.

## ⚙️ Funcionalidades Técnicas

* **Leitura Blindada de CSV:** Algoritmo robusto que ignora metadados (Data/Concurso) e foca apenas nas esferas sorteadas, compatível com separadores `;` e `,`.
* **Engenharia de Atributos (Feature Engineering):** A IA é treinada não apenas com os números, mas com dados derivados:
    * Distribuição de Dezenas (Quantos números nas casas 0-9, 10-19, etc.).
    * Soma, Pares, Primos, Fibonacci e Amplitude.
* **Ensemble Learning:** Média ponderada entre as previsões da LSTM e GRU.
* **Otimização Combinatória:** Gera um *Pool Expandido* de **30 dezenas** e processa mais de 140.000 combinações para filtrar as melhores baseadas em probabilidade.

## 🛠️ Tecnologias

* **Linguagem:** Python 3.8+
* **Core AI:** TensorFlow 2.x / Keras
* **Processamento de Dados:** Pandas, NumPy
* **Pré-processamento:** Scikit-Learn

## 🚀 Como Executar

Siga os passos abaixo para rodar o projeto no seu ambiente local (Windows/Linux/Mac).

### 1. Pré-requisitos
Certifique-se de ter o Python instalado. Recomenda-se o uso de um ambiente virtual.

### 2. Instalação

```bash
# 1. Clone o repositório
git clone [https://github.com/SEU_USUARIO/quina-ai.git](https://github.com/SEU_USUARIO/quina-ai.git)
cd quina-ai

# 2. Crie e ative o ambiente virtual (Recomendado)
python -m venv venv
# No Windows (Git Bash):
source venv/Scripts/activate
# No Windows (CMD):
venv\Scripts\activate

# 3. Instale as dependências
pip install numpy pandas tensorflow scikit-learn