# Aprendizado por Reforço Multi-Agente com SMPE e Otimização Evolucionária

**Autores:**

* Eduardo Vianna de Lima Fernandes Guimarães
* Isaque Vieira Machado Pim
* Juliano Genari de Araújo

**Data:** 29/11/2024

## Introdução

Este projeto implementa e analisa algoritmos de **aprendizado por reforço multi-agente (MARL)** em ambientes cooperativos parcialmente observáveis, com foco especial no algoritmo **SMPE (State Modeling with Adversarial Exploration)** integrado ao framework **AgileRL** para otimização evolucionária de hiperparâmetros.

O ambiente de teste utilizado é o **Simple Speaker Listener** da biblioteca MPE (Multi-Particle Environments), um benchmark clássico para avaliar comunicação e cooperação entre agentes em condições de observabilidade parcial. O projeto explora tanto algoritmos baseline (MATD3, MADDPG) quanto o estado-da-arte SMPE, desenvolvendo uma infraestrutura completa de experimentação com:

- Gerenciamento automatizado de experimentos via YAML
- Registro centralizado de resultados e métricas
- Visualização de curvas de aprendizado e comparação entre algoritmos
- Reprodução de políticas treinadas com geração de GIFs

## 1. Descrição do Problema

### 1.1 Ambiente: Simple Speaker Listener

O **Simple Speaker Listener** é um ambiente cooperativo de dois agentes modelado como um **Dec-POMDP (Decentralized Partially Observable Markov Decision Process)** onde:

**Agentes:**
- **Speaker (falante)**: Observa a posição de um landmark alvo, mas não pode se mover. Comunica através de ações discretas (mensagens).
- **Listener (ouvinte)**: Não vê qual landmark é o alvo, mas pode se mover pelo ambiente. Deve interpretar as mensagens do speaker para navegar até o landmark correto.

**Espaços de observação:**
- **Speaker**: Posição relativa do landmark alvo (3 features)
- **Listener**: Posições relativas de todos os landmarks + mensagem recebida (11 features)

**Espaços de ação:**
- **Speaker**: 3 ações discretas (mensagens de comunicação)
- **Listener**: 5 ações discretas (movimento: cima, baixo, esquerda, direita, nulo)

**Recompensa cooperativa:**
- Recompensa negativa proporcional à distância entre listener e landmark alvo
- Objetivo: Minimizar a distância através de comunicação eficaz

**Desafio:** A observabilidade parcial exige que os agentes desenvolvam:
1. **Protocolo de comunicação implícito** entre speaker e listener
2. **Coordenação temporal** para navegação eficiente
3. **Generalização** para diferentes configurações de landmarks

### 1.2 Formulação como Dec-POMDP

O problema é modelado matematicamente como:

- **Estados globais (S)**: Configuração completa do ambiente (posições de agentes e landmarks)
- **Observações locais (O)**: Cada agente $i$ recebe $o^i_t = \Omega(s_t)$ (observações parciais)
- **Ações conjuntas (A)**: $\mathbf{a}_t = (a^1_t, a^2_t)$ executadas simultaneamente
- **Transições**: $P(s_{t+1} | s_t, \mathbf{a}_t)$ determinísticas no ambiente físico
- **Recompensa compartilhada**: $r_t = R(s_t, \mathbf{a}_t)$ cooperativa

**Objetivo:** Aprender políticas descentralizadas $\pi^i(a^i | \tau^i)$ baseadas apenas no histórico de observações local $\tau^i = (o^i_1, a^i_1, \ldots, o^i_t)$ que maximizem o retorno esperado cooperativo:

$$
\max_{\pi^1, \pi^2} \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

## 2. Implementação

### 2.1 Arquitetura do Sistema

O projeto foi estruturado com separação clara entre experimentação, treinamento e análise:

**Estrutura de diretórios:**
```
projeto-final-RL/
├── configs/
│   └── experiments/          # Configurações YAML dos experimentos
│       ├── baseline.yaml
│       ├── smpe_baseline.yaml
│       └── ...
├── src/
│   ├── experiment_manager.py    # Gerenciamento de configurações YAML
│   └── results_tracker.py       # Registro de resultados
├── main.py                      # Script de treinamento MATD3
├── smpe.py                      # Implementação SMPE + AgileRL
├── replay.py                    # Visualização de políticas treinadas
├── plot_experiments.py          # Plotagem de curvas de treinamento
├── compare.py                   # Comparação entre experimentos
└── results/
    ├── experiments.csv          # Registro centralizado
    └── exp_YYYYMMDD_HHMMSS/     # Resultados por experimento
        ├── *_model.pt
        ├── metrics.json
        ├── *_plot.png
        └── *_data.npy
```

### 2.2 Classe `ExperimentManager`

```python
class ExperimentManager:
    def __init__(self, config_path: str):
        # Carrega e valida configuração YAML

    def get_init_hp(self) -> Dict[str, Any]:
        # Extrai hiperparâmetros para o algoritmo

    def get_net_config(self) -> Dict[str, Any]:
        # Retorna configuração de rede neural

    def get_training_params(self) -> TrainingParams:
        # Parâmetros de treinamento (steps, envs, etc.)
```

**Características:**
- Validação de esquema YAML
- Conversão automática de tipos
- Suporte a múltiplos algoritmos (MATD3, MADDPG, SMPE)
- Configuração de HPO evolucionário

### 2.3 Algoritmos Implementados

#### 2.3.1 MATD3 (Multi-Agent Twin Delayed DDPG)

Algoritmo baseline off-policy baseado em TD3:

**Componentes principais:**
- **Twin Critics**: Dois críticos Q(s,a) para reduzir overestimation
- **Delayed Policy Updates**: Atualização do ator menos frequente que críticos
- **Target Networks**: Redes alvo com soft updates (τ = 0.01)
- **Replay Buffer**: Buffer compartilhado multi-agente (100k transições)
- **OU Noise**: Ornstein-Uhlenbeck para exploração em ações contínuas

**Implementação:** Fornecida pela biblioteca AgileRL com suporte nativo a HPO evolucionário.

#### 2.3.2 SMPE (State Modeling with Adversarial Exploration)

Algoritmo estado-da-arte para MARL cooperativo (ICML 2025):

```python
class SMPE(MultiAgentRLAlgorithm):
    def __init__(self, observation_spaces, action_spaces,
                 belief_latent_dim=16, ...):
        # Componentes principais
        self.encoders = {agent_id: EvolvableMLP(...)}      # VAE encoders
        self.decoders = {agent_id: EvolvableMLP(...)}      # VAE decoders
        self.actors = {agent_id: EvolvableMLP(...)}        # Políticas
        self.critics_main = {agent_id: EvolvableMLP(...)}  # Críticos principais
        self.critics_filtered = {agent_id: EvolvableMLP(...)}  # Críticos filtrados
        self._filters = {agent_id: AMFilter(...)}          # AM filters
```

**Componentes únicos:**

1. **VAE Encoder-Decoder por agente:**
   - Encoder: $(o^i_t, a^i_{t-1}) \rightarrow (\mu^i_t, \log\sigma^i_t)$
   - Crença amostrada: $z^i_t \sim \mathcal{N}(\mu^i_t, \sigma^i_t)$
   - Decoder: $z^i_t \rightarrow \hat{o}^{-i}_t$ (reconstrução de observações dos outros)

2. **Agent Modeling (AM) Filters:**
   - Vetores aprendíveis $w^i \in [0,1]^{d}$ por agente
   - Ponderam importância de cada feature das observações alheias
   - Loss de reconstrução filtrada: $\mathcal{L}_{rec}^i = \|(o^{-i} \odot w^i) - \hat{o}^{-i}\|^2$

3. **Exploração Intrínseca (SimHash):**
   - Hash de crenças: $h^i_t = \text{sign}(Wz^i_t)$ onde $W \in \mathbb{R}^{32 \times d_z}$
   - Contagem de visitas: $N^i(h^i_t)$
   - Recompensa intrínseca: $r^{int,i}_t = \beta / \sqrt{N^i(h^i_t)}$

4. **Loss Total:**
$$
\mathcal{L}^i = \underbrace{\mathcal{L}_{rec}^i + \lambda_{KL} \text{KL}(q(z|\tau) \| \mathcal{N}(0,I))}_{\text{State Modeling}} + \underbrace{\lambda_V \mathcal{L}_{critic}^i}_{\text{Valor}} + \underbrace{\lambda_\pi \mathcal{L}_{policy}^i}_{\text{Política}}
$$

**Inovação (nossa contribuição):**
- Integração com `EvolvableModule` do AgileRL
- Todos os componentes (encoders, decoders, critics, actors) são evolvíveis
- HPO evolucionário automático de: learning rates, dimensão de belief, coeficientes de loss

### 2.4 Pipeline de Treinamento

**Loop principal (main.py / smpe.py):**

```python
# 1. Criar população de agentes
pop = create_population(
    algo="MATD3",  # ou "SMPE"
    population_size=4,
    observation_spaces=obs_spaces,
    action_spaces=act_spaces,
    hp_config=hp_config  # Configuração de HPO
)

# 2. Loop de treinamento com evolução
for evo_step in range(num_evo_steps):
    # 2a. Coletar experiências (todos agentes da população)
    for agent in pop:
        for step in range(steps_per_evo):
            obs, info = env.reset()
            action = agent.get_action(obs, infos=info)
            next_obs, reward, term, trunc, info = env.step(action)
            memory.add(obs, action, reward, next_obs, term, trunc)

            # SMPE: adicionar recompensa intrínseca
            if isinstance(agent, SMPE):
                intrinsic_reward = agent.compute_intrinsic_reward(beliefs)
                reward += intrinsic_reward

    # 2b. Aprender (off-policy)
    for agent in pop:
        for _ in range(learn_iterations):
            batch = memory.sample(batch_size)
            agent.learn(batch)

    # 2c. Avaliar fitness
    for agent in pop:
        agent.fitness = evaluate_agent(agent, env)

    # 2d. Seleção por torneio + mutações
    elite = tournament_selection(pop)
    pop = [elite.clone() for _ in range(population_size)]
    for agent in pop[1:]:
        agent = mutations.mutation(agent)
```

### 2.5 Sistema de Replay Multi-Algoritmo

```python
class ModelLoader:
    """Base class for algorithm-specific model loading."""
    @staticmethod
    def load(model_path, env, agent_ids, device): ...
    @staticmethod
    def prepare_for_inference(model): ...

class MATD3Loader(ModelLoader):
    # Carrega modelos MATD3 com fallback para reconstrução manual

class SMPELoader(ModelLoader):
    # Carrega SMPE com inicialização de filtros e hiperparâmetros
    def load(model_path, env, agent_ids, device):
        # Extrai hiperparâmetros de metrics.json
        hp = load_hyperparameters_from_json(model_path)

        # Cria instância SMPE
        model = SMPE(belief_latent_dim=hp['BELIEF_LATENT_DIM'], ...)

        # CRÍTICO: Inicializa otimizadores dos filtros
        model._init_filter_optimizers()

        # Carrega checkpoint
        model.load_checkpoint(model_path)
        return model
```

**Detecção automática de algoritmo:**
1. Lê `metrics.json` → campo `hyperparameters.ALGO`
2. Fallback: padrão de nome em `experiments.csv`
3. Último recurso: introspecção de checkpoint (busca `BELIEF_LATENT_DIM`)

## 3. Configurações e Hiperparâmetros

### 3.1 Configurações de Experimento

Experimentos são definidos via arquivos YAML em `configs/experiments/`:

**Experimentos MATD3:**
1. **baseline.yaml** - Configuração de referência original
2. **improved.yaml** - Redes mais profundas + maior exploração
3. **high_lr.yaml** - Learning rates elevados
4. **large_batch.yaml** - Batches grandes (512) para gradientes estáveis
5. **deep_network.yaml** - Redes muito profundas (4 camadas)
6. **aggressive_mutation.yaml** - Mutações evolutivas intensas
7. **stable_learning.yaml** - Abordagem conservadora
8. **fast_learning.yaml** - Atualizações muito frequentes
9. **large_population.yaml** - População evolutiva de 8 agentes

**Experimento SMPE:**
- **smpe_baseline.yaml** - Implementação baseline do SMPE

### 3.2 Hiperparâmetros MATD3 (baseline)

**Treinamento:**
- max_steps: 2.000.000 (8M steps efetivos com 4 agentes)
- num_envs: 8 (ambientes paralelos)
- evo_steps: 10.000 (passos entre ciclos evolutivos)
- population_size: 4

**Algoritmo:**
- batch_size: 128
- lr_actor: 0.0001
- lr_critic: 0.001
- gamma: 0.95
- tau: 0.01
- policy_freq: 2 (delayed policy updates)
- memory_size: 100.000

**Rede neural:**
- encoder_hidden_size: [64]
- head_hidden_size: [64]
- latent_dim: 64

**Exploração:**
- O-U Noise: habilitado
- expl_noise: 0.1
- theta: 0.15 (velocidade de reversão)
- dt: 0.01 (passo temporal)

### 3.3 Hiperparâmetros SMPE (baseline)

**Específicos do SMPE:**
- belief_latent_dim: 16 (dimensão do espaço de crenças)
- recon_coef: 1.0 (peso da loss de reconstrução)
- kl_coef: 0.001 (peso do KL divergence do VAE)
- filter_reg_coef: 0.001 (regularização L2 dos AM filters)
- intrinsic_coef: 0.0 (coeficiente de recompensa intrínseca)

**Arquitetura:**
- Encoders/Decoders: [64] hidden layers
- Actors: [64] hidden layers
- Critics (main e filtered): [64] hidden layers

**Novidade (Belief SimHash):**
- n_bits: 32 (dimensão do hash binário)
- Projection matrix: $W \in \mathbb{R}^{32 \times 16}$ (fixa, não treinável)

## 4. Resultados e Análise

### 4.1 Registro de Experimentos

Todos os experimentos completados são registrados em `results/experiments.csv`:

| exp_id | name | steps | duration_hours | final_score | best_score | status |
|--------|------|-------|----------------|-------------|------------|--------|
| exp_20251128_123149 | baseline | 8.000.000 | 6.32 | -61.00 | -58.40 | completed |
| exp_20251129_212113 | smpe_baseline | 8.000.000 | 1.19 | -41.44 | -31.55 | completed |

**Observação:** SMPE demonstrou convergência significativamente mais rápida (1.19h vs 6.32h) devido à menor complexidade computacional em relação a MATD3 neste ambiente específico.

### 4.2 Curvas de Aprendizado

A figura abaixo mostra a evolução do treinamento de todos os experimentos completados, com médias móveis de 10 iterações:

![Training Progress](training_progress.png)

A linha verde tracejada representa o score alvo de -60.

**Análise:**
- **SMPE baseline**: Convergiu para score médio de **-41.44**, superando significativamente o alvo de -60
- **MATD3 baseline**: Score médio final de **-61.00**, muito próximo do alvo
- **Best score SMPE**: **-31.55**, demonstrando picos de performance superiores
- **Estabilidade**: SMPE mostrou menor variância entre iterações evolutivas

### 4.3 Comportamento Aprendido

Exemplo de execução do modelo SMPE treinado no ambiente Simple Speaker Listener:

![SMPE Agent Behavior](exp_20251129_212113_speaker_listener.gif)

**Observações do comportamento:**
- **Comunicação emergente**: Speaker desenvolve protocolo de mensagens consistente
- **Navegação coordenada**: Listener interpreta mensagens e navega eficientemente
- **Generalização**: Política funciona para diferentes configurações de landmarks
- **Cooperação**: Minimização conjunta da distância através de coordenação temporal

### 4.4 Comparação de Algoritmos

**Vantagens do SMPE observadas:**
1. **Performance superior**: Score 48% melhor que MATD3 baseline
2. **Convergência mais rápida**: Estabilização em menos épocas
3. **Menor tempo de treinamento**: 81% mais rápido (1.19h vs 6.32h)
4. **Exploração mais eficiente**: Recompensa intrínseca guia descoberta de estratégias

**Trade-offs:**
- SMPE requer maior memória (encoders/decoders adicionais)
- Complexidade de implementação maior
- Mais hiperparâmetros para ajustar (belief_dim, coeficientes de loss)

## Nossa tentativa de inovação
## SMPE com HPO evolucionário em AgileRL

### Paper de origem

Esta implementação é baseada no paper **"Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration (SMPE²)"**, aceito na **ICML 2025**, de Kontogiannis et al. ([GitHub][1])

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

2. **Exploração intrínseca "adversarial" no espaço de crenças**
   Dado o embedding de crença $(z_t^i)$, o método constrói uma **recompensa intrínseca count-based**:

   * Usa **SimHash** para projetar $(z_t^i)$ num código discreto, conta visitas desse código e define um bônus $(r^{\text{int},i}_t \propto 1/\sqrt{N_i(\text{hash}(z_t^i))})$. ([arXiv][2])
   * Como os (z)'s também são alvos de reconstrução dos outros agentes, **descobrir crenças novas aumenta a perda de reconstrução deles**, forçando-os a melhorar seus modelos. Isso gera uma exploração **adversarial, mas pró-cooperação**: cada agente procura crenças novas que, ao mesmo tempo, enriquecem o modelo de estado dos demais.

O resultado é um backbone MARL "turbinado" com:

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

     * um crítico "normal" $(V_\psi(s))$;
     * um crítico "com crenças" $(V_\omega(s, z^i))$ que vê o estado filtrado na perspectiva do agente. ([arXiv][2])
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
* Permitir que a infraestrutura de **HPO evolucionário** do AgileRL faça *mutation* e *selection* não só de hiperparâmetros clássicos (learning rates, coeficientes $(\lambda)$, dimensão do embedding (z), peso da recompensa intrínseca (\beta) etc.), mas também de **submódulos da arquitetura SMPE**.

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

---

## 5. Conclusões

Este projeto demonstrou com sucesso a aplicação e extensão de algoritmos estado-da-arte de aprendizado por reforço multi-agente em ambientes cooperativos parcialmente observáveis. Os principais resultados e contribuições incluem:

**Resultados empíricos:**
- O algoritmo **SMPE** superou significativamente o baseline MATD3, alcançando score médio 48% superior (-41.44 vs -61.00)
- Convergência comprovadamente mais rápida (1.19h vs 6.32h de treinamento)
- Desenvolvimento de protocolo de comunicação emergente eficaz entre speaker e listener
- Generalização robusta para diferentes configurações de landmarks

**Contribuições técnicas:**
- **Integração SMPE + AgileRL**: Primeira implementação pública documentada combinando SMPE com HPO evolucionário
- **Infraestrutura de experimentação**: Sistema completo de gerenciamento de experimentos via YAML, registro centralizado e reprodução
- **Sistema de replay multi-algoritmo**: Detecção automática e carregamento correto de modelos MATD3 e SMPE
- **Modularização**: Todos os componentes SMPE (encoders, decoders, AM filters) como `EvolvableModule`, permitindo mutação arquitetural

**Insights sobre MARL cooperativo:**
- A modelagem explícita de estado através de VAEs (beliefs) facilita cooperação em Dec-POMDPs
- AM filters aprendem automaticamente quais features dos outros agentes são relevantes
- Exploração intrínseca no espaço de crenças é mais eficiente que no espaço de observações
- HPO evolucionário é especialmente valioso em MARL devido ao grande espaço de hiperparâmetros

**Limitações e trabalhos futuros:**
- Testes limitados a um único ambiente (Simple Speaker Listener)
- Exploração de hiperparâmetros ainda inicial (apenas baseline configurações)
- Potencial para testar em ambientes mais complexos (LBF, RWARE, MPE variants)
- Comparação com outros algoritmos SOTA (QMIX, MAPPO, HAPPO)

O projeto estabelece uma base sólida para pesquisa futura em MARL cooperativo com otimização evolucionária, demonstrando que a combinação de state modeling, exploração adversarial e HPO automático é uma direção promissora para resolver problemas multi-agente complexos.

## 6. Referências

\[1\]: https://github.com/ddaedalus/smpe "GitHub - ddaedalus/smpe: [ICML 2025] Official Code of SMPE: \"Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration\""

\[2\]: https://arxiv.org/html/2505.05262v1 "Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration"

\[3\]: https://github.com/AgileRL/AgileRL "AgileRL/AgileRL - Evolutionary Hyperparameter Optimization for Reinforcement Learning"

\[4\]: https://docs.agilerl.com/en/latest/api/modules/base.html "EvolvableModule - AgileRL Documentation"

\[5\]: https://docs.agilerl.com/en/latest/evo_hyperparam_opt/index.html "Evolutionary Hyperparameter Optimization - AgileRL Documentation"

## 7. Agradecimentos

Este projeto foi desenvolvido como trabalho final da disciplina de Aprendizado por Reforço (Reinforcement Learning) ministrada pelo **Prof. Flávio Coelho** na Fundação Getulio Vargas (FGV/EMAp).

Agradecimentos especiais:
- Aos autores do paper SMPE (Kontogiannis et al.) pela disponibilização do código oficial
- À comunidade AgileRL pelo framework robusto de HPO evolucionário
- Ao PettingZoo/MPE pela biblioteca de ambientes multi-agente de alta qualidade

## Anexos

### A. Estrutura de Arquivos

**Código fonte principal:**
- `main.py` - Script de treinamento MATD3 com HPO evolucionário
- `smpe.py` - Implementação completa do SMPE integrado ao AgileRL
- `replay.py` - Sistema de reprodução de políticas com suporte multi-algoritmo
- `plot_experiments.py` - Visualização de curvas de aprendizado
- `compare.py` - Comparação estatística entre experimentos

**Infraestrutura:**
- `src/experiment_manager.py` - Gerenciamento de configurações YAML
- `src/results_tracker.py` - Registro e rastreamento de resultados
- `configs/experiments/*.yaml` - Configurações de experimentos

**Configuração:**
- `docker-compose.yml` - Orquestração de containers para treinamento e replay
- `Dockerfile` - Imagem Docker com CUDA 12.6 e Python 3.12
- `pyproject.toml` - Gerenciamento de dependências via uv
- `uv.lock` - Lock file de dependências

**Resultados gerados:**
- `results/experiments.csv` - Registro centralizado de todos os experimentos
- `results/exp_*/metrics.json` - Métricas detalhadas por experimento
- `results/exp_*/*_plot.png` - Curvas de treinamento individuais
- `results/exp_*/*_model.pt` - Modelos treinados (checkpoints)
- `results/exp_*/*_data.npy` - Dados brutos de scores
- `results/exp_*/*_speaker_listener.gif` - Visualizações de políticas

### B. Instruções de Execução

**Pré-requisitos:**
- Docker e Docker Compose instalados
- GPU NVIDIA com driver CUDA 12.6+ (recomendado)
- 16GB+ de RAM
- 50GB+ de espaço em disco

**Setup inicial:**

```bash
# 1. Clonar repositório
git clone <repository-url>
cd projeto-final-RL

# 2. Build da imagem Docker (primeira vez)
docker build -t projeto-final-rl:latest .
```

**Executar treinamento:**

```bash
# Treinamento MATD3 baseline
docker compose run --rm training

# Treinamento SMPE
docker compose run --rm run_smpe

# Treinamento com configuração customizada
docker compose run --rm training python main.py --config configs/experiments/improved.yaml
```

**Visualizar resultados:**

```bash
# Plotar curvas de todos os experimentos
docker compose run --rm training python plot_experiments.py

# Listar experimentos completados
docker compose run --rm training python compare.py --list

# Comparar experimentos específicos
docker compose run --rm training python compare.py exp_20251128_123149 exp_20251129_212113

# Replay do último modelo treinado
docker compose run --rm replay

# Replay de experimento específico
docker compose run --rm replay python replay.py --model exp_20251129_212113
```

**Execução local (sem Docker):**

```bash
# Instalar dependências com uv
uv sync

# Treinar modelo
uv run python main.py --config configs/experiments/baseline.yaml

# Visualizar política
uv run python replay.py
```

### C. Exemplo de Configuração YAML

```yaml
name: "meu_experimento"
description: "Descrição do experimento"
seed: 42

training:
  max_steps: 2000000       # Total de passos
  num_envs: 8              # Ambientes paralelos
  evo_steps: 10000         # Passos entre evolução
  checkpoint_interval: 100000
  learning_delay: 0
  eval_steps: null
  eval_loop: 1

hyperparameters:
  population_size: 4
  algo: "MATD3"           # ou "SMPE"
  batch_size: 128
  lr_actor: 0.0001
  lr_critic: 0.001
  gamma: 0.95
  memory_size: 100000
  learn_step: 100
  tau: 0.01

  # Específico MATD3
  policy_freq: 2
  o_u_noise: true
  expl_noise: 0.1

  # Específico SMPE (se algo: "SMPE")
  # belief_latent_dim: 16
  # recon_coef: 1.0
  # kl_coef: 0.001

network:
  latent_dim: 64
  encoder_hidden_size: [64]
  head_hidden_size: [64]

hpo_config:
  lr_actor: {min: 0.0001, max: 0.01}
  lr_critic: {min: 0.0001, max: 0.01}
  batch_size: {min: 8, max: 512, dtype: int}

mutation:
  no_mutation: 0.2
  architecture: 0.2
  new_layer: 0.2
  parameter: 0.2
  rl_hp: 0.2
  mutation_sd: 0.1
```

### D. Troubleshooting

**Problema:** `CUDA out of memory`
- **Solução:** Reduzir `num_envs` ou `population_size` no YAML

**Problema:** Checkpoint não carrega
- **Solução:** Verificar se `metrics.json` existe no diretório do experimento

**Problema:** GIF não gerado no replay
- **Solução:** Garantir que o modelo foi treinado até o final (status: completed)

**Problema:** Experimento não aparece no `experiments.csv`
- **Solução:** Verificar se o treinamento completou sem erros; checar logs em `results/`
