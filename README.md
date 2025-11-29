# Projeto de Eduardo Vianna de Lima Fernandes Guimarães, Isaque Vieira Machado Pim e Juliano Genari de Araújo:

## Como Usar

### Opção 1: Docker Compose (Recomendado)

A forma mais fácil de executar o projeto é usando Docker Compose:

**1. Build da imagem (apenas na primeira vez ou após mudar dependências):**
```bash
docker build -t projeto-final-rl:latest .
```

**2. Executar o treinamento:**
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

# Nossa tentativa de inovação
## SMPE com HPO evolucionário em AgileRL

### Paper de origem

Esta implementação é baseada no paper **“Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration (SMPE²)”**, aceito na **ICML 2025**, de Kontogiannis et al. ([GitHub][1])

O trabalho trata de **multi-agent deep reinforcement learning (MARL)** em ambientes cooperativos modelados como **Dec-POMDPs** (parcialmente observáveis, sem canal explícito de comunicação). O foco é:

* Aprender **representações de estado (beliefs)** a partir das observações locais de cada agente.
* Usar essas crenças para **melhorar cooperação e exploração** via *intrinsic motivation*.
* O método **SMPE/SMPE²** supera algoritmos MARL SOTA em tarefas cooperativas dos benchmarks **MPE, LBF e RWARE**. ([GitHub][1])

---

### Ideia por trás do SMPE / SMPE²

A ideia central do SMPE é adicionar, em cima de um algoritmo MARL padrão (no paper, **MAA2C**), duas camadas extras:

1. **State modelling auto-supervisionado**
   Cada agente (i) aprende um modelo que, a partir do seu histórico local $(\tau_t^i)$, infere uma **variável latente de crença** $(z_t^i)$ sobre o estado global não observado.

   * Isso é feito com um **encoder–decoder variacional (VAE)** que tenta reconstruir **partes informativas** das observações dos outros agentes usando apenas a observação própria. ([arXiv][2])
   * Para evitar redundância, o paper introduz os **Agent Modelling (AM) filters**: vetores de pesos $(w^i\in[0,1]^d)$ que dizem quanto cada feature dos outros agentes é relevante para o agente (i).

2. **Exploração intrínseca “adversarial” no espaço de crenças**
   Dado o embedding de crença $(z_t^i)$, o método constrói uma **recompensa intrínseca count-based**:

   * Usa **SimHash** para projetar $(z_t^i)$ num código discreto, conta visitas desse código e define um bônus $(r^{\text{int},i}_t \propto 1/\sqrt{N_i(\text{hash}(z_t^i))})$. ([arXiv][2])
   * Como os (z)’s também são alvos de reconstrução dos outros agentes, **descobrir crenças novas aumenta a perda de reconstrução deles**, forçando-os a melhorar seus modelos. Isso gera uma exploração **adversarial, mas pró-cooperação**: cada agente procura crenças novas que, ao mesmo tempo, enriquecem o modelo de estado dos demais.

O resultado é um backbone MARL “turbinado” com:

* crenças latentes **relevantes para a política** (não só para reconstrução), e
* uma política de exploração guiada nesses beliefs, em vez de diretamente no espaço de observações.

---

### Etapas do algoritmo

Resumindo o **Algoritmo SMPE/SMPE²**: ([arXiv][2])

1. Inicializar:

   * Actors e Critics do backbone MARL.
   * Encoders–decoders variacionais por agente (para o state modelling).
   * AM filters $(w^i)$ e suas redes.
   * *Replay buffer* compartilhado.

2. **Coleta de dados**
   Para cada passo de tempo:

   * Cada agente recebe observação local $(o_t^i)$.
   * Amostra crença $(z_t^i \sim q_\phi^i(z\mid\tau_t^i))$.
   * Amostra ação $(a_t^i \sim \pi_\theta^i(a\mid o_t^i, z_t^i))$.
   * Executa ações conjuntas, recebe recompensa global $(r_t)$ e novas observações.

3. **Cálculo de recompensa intrínseca**

   * Para cada agente, calcula hash de $(z_t^i)$ via SimHash, atualiza contagem e computa
  
     $( r^{\text{int},i}_t = \frac{\beta}{\sqrt{N_i(\text{hash}(z_t^i))}}.)$
     
   * Define a recompensa total usada no update de política como
     $(\tilde r_t^i = r_t + r^{\text{int},i}_t)$.

4. **Armazenamento**

   * Armazena transições $((o_t, a_t, \tilde r_t, o_{t+1}, \ldots))$ no *buffer*.

5. **Atualização do state modelling (VAE + AM filters)**
   Periodicamente, para cada agente:

   * Minimiza a **loss de reconstrução filtrada**
     $L*{rec}^i
     = E*{q_\phi^i(z\mid\tau)}
     \big| (o^{-i} \odot w^i) - \hat o^{-i}(z)\big|^2,$
     onde $(o^{-i})$ são observações dos outros agentes e $(\odot)$ é produto elemento a elemento. ([arXiv][2])
   * Adiciona:

     * termo **KL** para o VAE: $(\mathrm{KL}\big(q_\phi^i(z\mid\tau),|,\mathcal{N}(0,I)\big))$;
     * regularização L2 nos filtros $(w^i)$ para evitar colapso em zero.
   * A combinação desses termos define a loss de **state modelling** $(\mathcal{L}_{\text{SM}}^i)$.

6. **Atualização dos críticos e da política**

   * Mantém dois críticos:

     * um crítico “normal” $(V_\psi(s))$;
     * um crítico “com crenças” $(V_\omega(s, z^i))$ que vê o estado filtrado na perspectiva do agente. ([arXiv][2])
   * Minimiza erros TD correspondentes (loss de valor).
   * Atualiza o ator com gradiente de política usando estados estendidos $((o^i, z^i))$.
   * A loss total por agente combina:
     $(L^i = L_{SM}^i \cdot \lambda_V L_{crit}^i - \lambda_\pi J_{pol}^i.)$

7. **Loop até o fim do treinamento**, com updates periódicos de *target networks* e dos encoders–decoders para estabilizar o cálculo dos bônus intrínsecos.

Matematicamente, o paper mostra que o problema de **state modelling** é formulado de forma que o valor ótimo com beliefs $(z)$ é equivalente ao valor ótimo do Dec-POMDP original (Proposição 2.1), ou seja, introduzir $(z)$ **não restringe** o conjunto de políticas ótimas possíveis – ele só reparametriza o problema de forma mais informativa. ([arXiv][2])

---

### Nossa contribuição: SMPE + HPO evolucionário em AgileRL

A implementação deste repositório parte do **código oficial do SMPE/SMPE²** ([GitHub][1]) e o adapta para o ecossistema da biblioteca **[AgileRL](https://github.com/AgileRL/AgileRL)**, que é focada em **hiperparameter optimization (HPO) evolucionário** para RL. ([GitHub][3])

A principal ideia é:

* **Encapsular os componentes centrais do SMPE** (encoder–decoder de state modelling, AM filters, críticos adicionais, cálculo de bônus intrínseco) como um **`EvolvableModule`** do AgileRL. ([docs.agilerl.com][4])
* Permitir que a infraestrutura de **HPO evolucionário** do AgileRL faça *mutation* e *selection* não só de hiperparâmetros clássicos (learning rates, coeficientes $(\lambda)4, dimensão do embedding (z), peso da recompensa intrínseca (\beta) etc.), mas também de **submódulos da arquitetura SMPE**.

Em termos práticos:

* Os objetos do modelo foram reescritos herdando de `EvolvableModule`, ganhando suporte nativo a:

  * `.to(device)` integrado com o restante da lib;
  * registro automático de métodos de mutação;
  * participação transparente na **população evolutiva** usada para HPO.
* A nossa **versão de SMPE** é tratada como mais um algoritmo multi-agente do AgileRL, podendo:

  * ser instanciada em **populações** de agentes;
  * ser otimizada por **seleção de torneio + mutações** com base no retorno obtido, como descrito na documentação de HPO evolucionário da lib. ([docs.agilerl.com][5])

> **Inovação**
> A nossa contribuição principal é **combinar o SMPE com HPO evolucionário do AgileRL**, expondo os módulos de *state modelling* (incluindo os AM filters) como entidades evolutivas. Até onde sabemos, ainda não há na literatura uma integração pública de **SMPE + HPO evolucionário em AgileRL**, o que torna este repositório um primeiro passo nessa direção.

Isso significa que:

* o ajuste fino dos hiperparâmetros sensíveis do SMPE é automatizado;
* o algoritmo pode ser reusado em novos cenários (por exemplo, diferentes tarefas da MPE, LBF, RWARE ou outros ambientes PettingZoo) sem precisar redesenhar o tuning do zero;
* ganha-se um pipeline mais próximo de uso real: um único experimento com população evolutiva encontra tanto a política SMPE quanto uma configuração razoável de hiperparâmetros.

[1]: https://github.com/ddaedalus/smpe "GitHub - ddaedalus/smpe: [ICML 2025] Official Code of SMPE: \"Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration\""
[2]: https://arxiv.org/html/2505.05262v1 "Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration"
[3]: https://github.com/AgileRL/AgileRL?utm_source=chatgpt.com "AgileRL/AgileRL"
[4]: https://docs.agilerl.com/en/latest/api/modules/base.html?utm_source=chatgpt.com "EvolvableModule"
[5]: https://docs.agilerl.com/en/latest/evo_hyperparam_opt/index.html?utm_source=chatgpt.com "Evolutionary Hyperparameter Optimization"


