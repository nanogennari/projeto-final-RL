# Ambiente Multi-Agente Speaker-Listener

Este projeto implementa um ambiente de aprendizado por reforço multi-agente baseado no [ambiente Speaker-Listener](https://pettingzoo.farama.org/environments/mpe/simple_speaker_listener), onde dois agentes colaboram para resolver tarefas de comunicação e coordenação.   Os Agentes são treinados usando o Algoritmo MATD3.   
Esta [implementação](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/matd3.html#matd3-tutorial) é fornecida pelo pacote AgileRL, sem garantias de performance.

## Visão Geral

O ambiente Speaker-Listener consiste em dois agentes com papéis distintos:

- **Speaker (Falante)**: Fala mas não pode se mover.
- **Listener (Ouvinte)**: Ouve as mensagens do Speaker e precisa navegar até O alvo.

Um descrição detalhada deste ambiente pode ser encontrada [neste artigo](https://arxiv.org/pdf/1706.02275)
                              
## Características
- Ambiente colaborativo onde os agentes devem aprender a se comunicar eficazmente                     
- Treinamento usando algoritmos de aprendizado por reforço multi-agente
- Suporte para GPU via PyTorch CUDA         
- Implementação modular e extensível 
## Requisitos
- Docker e Docker Compose (recomendado)
- GPU NVIDIA com CUDA (para aceleração GPU)
- **OU** Python 3.12+ com **uv** (para execução local)

## Como Usar

### Opção 1: Docker Compose (Recomendado)

A forma mais fácil de executar o projeto é usando Docker Compose:

**1. Build da imagem (apenas na primeira vez ou após mudar dependências):**
```bash
docker build -t projeto-final-rl:latest .
```

**2. Executar o treinamento:**
```bash
docker compose run --rm training
```

**3. Visualizar o modelo treinado:**
```bash
docker compose run --rm replay
```

### Opção 2: Execução Local com uv

Após copiar este diretório localmente, inicialize o ambiente virtual definido em pyproject.toml:

```bash
uv sync
```

Execute o script principal:
```bash
python main.py
```

Para gerar a visualização do modelo treinado:

```bash
python replay.py
```

## Organização dos Modelos

O sistema detectará automaticamente se CUDA está disponível e utilizará GPU quando possível.

**Cada execução de treinamento salva automaticamente:**

```
models/MATD3/
├── 20251127_143052/              # Timestamped run
│   ├── MATD3_trained_agent.pt    # Modelo treinado
│   ├── hyperparameters.txt       # Todos os hiperparâmetros usados
│   ├── training_scores_evolution.png  # Gráfico de evolução
│   └── training_scores_history.npy    # Dados brutos
├── 20251127_190234/              # Outra execução
│   └── ...
└── latest -> 20251127_190234/    # Symlink para o modelo mais recente
```

- Cada treinamento cria um diretório com timestamp único
- Os hiperparâmetros são salvos em `hyperparameters.txt` para comparação
- O link `latest/` sempre aponta para o modelo mais recente
- `replay.py` carrega automaticamente o modelo mais recente



## Tarefa

Sua tarefa é implementar um novo algoritmo de aprendizado por reforço multi-agente para o ambiente Speaker-Listener. Este algoritmo deve ser capaz de fazer com que o listener consiga navegar até o alvo mais rápido do que o algoritmo [MATD3 original](https://docs.agilerl.com/en/latest/api/algorithms/matd3.html), ou seja, consiga alcançar um score médio maior que -60(score médio da configuração atual). Alternativamente você pode tentar melhorar a configuração do algoritmo atual de forma a superar  a performance atual.

Para saber mais sobre o algoritmo MATD3, consulte [este artigo](https://arxiv.org/abs/1910.01465).


