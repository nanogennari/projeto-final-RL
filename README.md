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
- **Sistema de gerenciamento de experimentos** com checkpoint/resume
- **Monitoramento de progresso em tempo real** via CSV
- **Configurações YAML** para fácil experimentação
- **Comparação de resultados** entre diferentes experimentos 
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

**2. Executar o treinamento com configuração específica:**
```bash
# Usar configuração baseline (padrão)
docker compose run --rm training

# Ou especificar uma configuração customizada
docker compose run --rm training python main.py --config configs/experiments/improved.yaml
```

**3. Monitorar o progresso durante o treinamento:**

Em outro terminal, enquanto o treinamento está rodando:
```bash
# Ver resumo do progresso atual
python summary.py

# Ou monitorar continuamente (atualiza a cada 10 segundos)
watch -n 10 python summary.py
```

**4. Retomar treinamento interrompido:**

Se o treinamento for interrompido, ele será retomado automaticamente do último checkpoint:
```bash
docker compose run --rm training
```

**5. Visualizar o modelo treinado:**
```bash
docker compose run --rm replay
```

**6. Comparar resultados de diferentes experimentos:**
```bash
# Listar todos os experimentos
python compare.py --list

# Comparar dois experimentos específicos
python compare.py exp_20251127_143052 exp_20251127_190234
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

## Estrutura do Projeto

O sistema detectará automaticamente se CUDA está disponível e utilizará GPU quando possível.

### Diretórios

```
projeto-final-RL/
├── configs/
│   └── experiments/
│       ├── baseline.yaml        # Configuração baseline MATD3
│       └── improved.yaml        # Configuração otimizada
├── src/                         # Módulos de gerenciamento
│   ├── checkpoint_manager.py   # Salvar/carregar checkpoints
│   ├── experiment_manager.py   # Carregar configs YAML
│   ├── progress_tracker.py     # Monitoramento em tempo real
│   └── results_tracker.py      # Registro de experimentos
├── checkpoints/                 # Checkpoints temporários (não versionado)
│   └── exp_YYYYMMDD_HHMMSS/    # Checkpoints por experimento
├── progress/                    # CSVs de progresso (não versionado)
│   └── exp_YYYYMMDD_HHMMSS.csv # Progresso em tempo real
├── results/                     # Resultados finais
│   ├── experiments.csv          # Registro centralizado
│   └── exp_YYYYMMDD_HHMMSS/    # Resultados por experimento
│       ├── model.pt            # Modelo final
│       ├── metrics.json        # Métricas detalhadas
│       ├── scores_plot.png     # Gráfico de evolução
│       └── scores_data.npy     # Dados brutos
├── models/                      # (Legado - mantido para compatibilidade)
└── videos/                      # Vídeos de replay
```

### Configurações YAML

As configurações de experimentos são definidas em arquivos YAML em `configs/experiments/`.

**Experimentos Disponíveis**:

1. **baseline.yaml** - Configuração original de referência
2. **improved.yaml** - Melhorias balanceadas (redes mais profundas, maior exploração)
3. **high_lr.yaml** - Learning rates altos para aprendizado rápido
4. **large_batch.yaml** - Batches grandes para gradientes estáveis
5. **deep_network.yaml** - Redes muito profundas para aprendizado complexo
6. **aggressive_mutation.yaml** - Mutações evolutivas agressivas
7. **stable_learning.yaml** - Abordagem conservadora e estável
8. **fast_learning.yaml** - Atualizações muito frequentes
9. **large_population.yaml** - População grande para diversidade evolutiva

Veja `configs/experiments/README.md` para detalhes completos de cada configuração.

**Exemplo de Configuração**:

```yaml
# configs/experiments/baseline.yaml
name: "baseline_matd3"
description: "MATD3 baseline configuration"
seed: 42

training:
  max_steps: 2000000
  num_envs: 8
  evo_steps: 10000
  checkpoint_interval: 100000  # Salva checkpoint a cada 100k steps
  learning_delay: 0
  eval_steps: null
  eval_loop: 1

hyperparameters:
  population_size: 4
  batch_size: 128
  lr_actor: 0.0001
  lr_critic: 0.001
  gamma: 0.95
  memory_size: 100000
  learn_step: 100
  tau: 0.01

network:
  latent_dim: 64
  encoder_hidden_size: [64]
  head_hidden_size: [64]

hpo_config:  # Hyperparameter optimization ranges
  lr_actor:
    min: 0.0001
    max: 0.01
  lr_critic:
    min: 0.0001
    max: 0.01

mutation:  # Evolutionary mutation probabilities
  no_mutation: 0.2
  architecture: 0.2
  new_layer: 0.2
  parameter: 0.2
  rl_hp: 0.2
```

**Para criar uma nova configuração**:
```bash
cp configs/experiments/baseline.yaml configs/experiments/my_experiment.yaml
# Edite my_experiment.yaml conforme necessário
```

### Sistema de Checkpoints

O sistema salva checkpoints automáticos durante o treinamento:

- **Intervalo**: Configurável via `checkpoint_interval` (padrão: 100.000 steps)
- **Conteúdo**: População de agentes, replay buffer, estados RNG, metadados
- **Localização**: `checkpoints/exp_YYYYMMDD_HHMMSS/`
- **Retenção**: Mantém apenas os últimos 3 checkpoints para economizar espaço
- **Resume automático**: Ao reiniciar o treinamento, continua do último checkpoint automaticamente
- **Limpeza**: Checkpoints são removidos após conclusão bem-sucedida do treinamento

### Resultados e Registro de Experimentos

Cada experimento finalizado é registrado em `results/experiments.csv` com:

- **exp_id**: ID único do experimento
- **name**: Nome da configuração
- **status**: running, completed, failed
- **steps**: Total de steps treinados
- **duration_hours**: Duração total do treinamento
- **final_score**: Score médio final
- **best_score**: Melhor score obtido
- **worst_score**: Pior score obtido
- **config_path**: Caminho para o arquivo de configuração YAML
- **start_time**: Data/hora de início
- **end_time**: Data/hora de conclusão

Resultados detalhados por experimento em `results/exp_YYYYMMDD_HHMMSS/`:

- `model.pt`: Modelo elite final treinado
- `metrics.json`: Métricas completas do experimento
- `scores_plot.png`: Gráfico da evolução das pontuações
- `scores_data.npy`: Dados brutos das pontuações para análise

## Workflows Comuns

### Executar um Novo Experimento

1. **Criar ou escolher uma configuração**:
   ```bash
   # Usar baseline existente
   config="configs/experiments/baseline.yaml"

   # Ou criar nova configuração
   cp configs/experiments/baseline.yaml configs/experiments/my_experiment.yaml
   # Editar my_experiment.yaml conforme necessário
   ```

2. **Iniciar o treinamento**:
   ```bash
   docker compose run --rm training python main.py --config $config
   ```

3. **Monitorar em tempo real** (em outro terminal):
   ```bash
   # Ver resumo atual
   python summary.py

   # Ou monitorar continuamente
   watch -n 10 python summary.py
   ```

### Executar Múltiplos Experimentos

**Rodar todos os experimentos automaticamente**:
```bash
./run_all_experiments.sh
```

**Ou rodar manualmente**:
```bash
# Executar sequência de experimentos (Fase 1: Testes Rápidos)
for config in baseline high_lr fast_learning; do
  echo "===== Iniciando experimento: $config ====="
  docker compose run --rm training python main.py --config configs/experiments/${config}.yaml
  echo "===== Experimento $config concluído ====="
done

# Executar em background e continuar depois
nohup docker compose run --rm training python main.py --config configs/experiments/deep_network.yaml > deep_network.log 2>&1 &

# Monitorar experimento em background
tail -f deep_network.log
```

**Estratégia Recomendada** (veja `configs/experiments/README.md` para detalhes):

**Fase 1** - Testes Rápidos (~16h total):
```bash
# Estabelecer baseline e testar aprendizado rápido
docker compose run --rm training python main.py --config configs/experiments/baseline.yaml
docker compose run --rm training python main.py --config configs/experiments/high_lr.yaml
docker compose run --rm training python main.py --config configs/experiments/fast_learning.yaml
```

**Fase 2** - Variações de Arquitetura (~20h total):
```bash
# Testar melhorias arquiteturais
docker compose run --rm training python main.py --config configs/experiments/improved.yaml
docker compose run --rm training python main.py --config configs/experiments/deep_network.yaml
docker compose run --rm training python main.py --config configs/experiments/large_batch.yaml
```

**Fase 3** - Estratégias Avançadas (~23h total):
```bash
# Busca evolutiva e diversidade populacional
docker compose run --rm training python main.py --config configs/experiments/aggressive_mutation.yaml
docker compose run --rm training python main.py --config configs/experiments/stable_learning.yaml
docker compose run --rm training python main.py --config configs/experiments/large_population.yaml
```

### Retomar Treinamento Interrompido

Se o treinamento for interrompido (Ctrl+C, falta de energia, etc.), simplesmente execute novamente:

```bash
docker compose run --rm training python main.py --config configs/experiments/baseline.yaml
```

O sistema irá:
1. Detectar automaticamente o checkpoint mais recente
2. Carregar estado completo (população, replay buffer, RNG states)
3. Continuar do exato ponto onde parou

### Comparar Resultados de Experimentos

1. **Listar todos os experimentos**:
   ```bash
   python compare.py --list
   ```

2. **Comparar dois ou mais experimentos**:
   ```bash
   python compare.py exp_20251127_143052 exp_20251127_190234
   ```

   Saída exemplo:
   ```
   ==================================================================================================
   EXPERIMENT COMPARISON
   ==================================================================================================

   Metric                   baseline_matd3           improved_matd3
   --------------------------------------------------------------------------------------------------
   Experiment ID            exp_20251127_143052      exp_20251127_190234
   Status                   completed                completed
   Steps                    2000000                  2000000
   Duration (hours)         5.8                      6.2
   Final Score              -58.2                    -52.7
   Best Score               -45.3                    -38.9
   Worst Score              -72.1                    -65.4

   COMPARISON:
     Score Difference: +5.5
     Percentage Change: +9.45%
     ✓ improved_matd3 is BETTER
   ```

3. **Analisar métricas detalhadas**:
   ```bash
   cat results/exp_20251127_143052/metrics.json | python -m json.tool
   ```

### Visualizar Modelo Treinado

```bash
# Visualizar o modelo mais recente
docker compose run --rm replay

# Ou especificar um experimento específico
docker compose run --rm replay python replay.py --model results/exp_20251127_143052/model.pt
```

### Limpeza de Espaço em Disco

Durante experimentos longos, checkpoints podem ocupar espaço. O sistema automaticamente:

- Mantém apenas os últimos 3 checkpoints durante o treinamento
- Remove todos os checkpoints após conclusão bem-sucedida

Para limpeza manual:
```bash
# Remover checkpoints de treinamentos antigos
rm -rf checkpoints/

# Remover CSVs de progresso antigos
rm -rf progress/

# Manter apenas o registro central de experimentos
find results/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;
# CUIDADO: Isso remove todos os modelos e gráficos salvos!
```

## Tarefa

Sua tarefa é implementar um novo algoritmo de aprendizado por reforço multi-agente para o ambiente Speaker-Listener. Este algoritmo deve ser capaz de fazer com que o listener consiga navegar até o alvo mais rápido do que o algoritmo [MATD3 original](https://docs.agilerl.com/en/latest/api/algorithms/matd3.html), ou seja, consiga alcançar um score médio maior que -60(score médio da configuração atual). Alternativamente você pode tentar melhorar a configuração do algoritmo atual de forma a superar  a performance atual.

Para saber mais sobre o algoritmo MATD3, consulte [este artigo](https://arxiv.org/abs/1910.01465).


