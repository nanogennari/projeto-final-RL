# Ambiente Multi-Agente Speaker-Listener

Este projeto implementa um ambiente de aprendizado por reforço multi-agente baseado no [ambiente Speaker-Listener](https://pettingzoo.farama.org/environments/mpe/simple_speaker_listener), onde dois agentes colaboram para resolver tarefas de comunicação e coordenação.   Os Agentes são treinados usando o Algoritmo MATD3.
Esta [implementação](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/matd3.html#matd3-tutorial) é fornecida pelo pacote AgileRL, sem garantias de performance.

## Visão Geral

O ambiente Speaker-Listener consiste em dois agentes com papéis distintos:

- **Speaker (Falante)**: Fala mas não pode se mover.
- **Listener (Ouvinte)**: Ouve as mensagens do Speaker e precisa navegar até O alvo.

Um descrição detalhada deste ambiente pode ser encontrada [neste artigo](https://arxiv.org/pdf/1706.02275)

** Executar o treinamento com configuração específica:**
```bash
# Usar configuração baseline (padrão)
docker compose run --rm training

# Ou especificar uma configuração customizada
docker compose run --rm training python main.py --config configs/experiments/improved.yaml
```

** Comparar resultados de diferentes experimentos:**
# Listar todos os experimentos
python compare.py --list

# Comparar dois experimentos específicos
python compare.py exp_20251127_143052 exp_20251127_190234

### Configurações YAML dos experimentos

As configurações estão em `configs/experiments/`.

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
name: "baseline_matd3"
description: "MATD3 baseline configuration"
seed: 42

training:
  max_steps: 2000000
  num_envs: 8
  evo_steps: 10000
  checkpoint_interval: 100000 
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

hpo_config:
  lr_actor:
    min: 0.0001
    max: 0.01
  lr_critic:
    min: 0.0001
    max: 0.01

mutation:
  no_mutation: 0.2
  architecture: 0.2
  new_layer: 0.2
  parameter: 0.2
  rl_hp: 0.2
```

**Para criar uma nova configuração**:
```bash
cp configs/experiments/baseline.yaml configs/experiments/my_experiment.yaml
```

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
   # baseline existente
   config="configs/experiments/baseline.yaml"

   # criar nova configuração
   cp configs/experiments/baseline.yaml configs/experiments/my_experiment.yaml
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
for config in baseline high_lr fast_learning; do
  echo "===== Iniciando experimento: $config ====="
  docker compose run --rm training python main.py --config configs/experiments/${config}.yaml
  echo "===== Experimento $config concluído ====="
done
nohup docker compose run --rm training python main.py --config configs/experiments/deep_network.yaml > deep_network.log 2>&1 &

# Monitorar experimento em background
tail -f deep_network.log
```

### Visualizar Modelo Treinado

```bash
# Visualizar o modelo mais recente
docker compose run --rm replay

# Ou especificar um experimento específico
docker compose run --rm replay python replay.py --model results/exp_20251127_143052/model.pt
```

