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
   Cada agente (i) aprende um modelo que, a partir do seu histórico local (\tau_t^i), infere uma **variável latente de crença** (z_t^i) sobre o estado global não observado.

   * Isso é feito com um **encoder–decoder variacional (VAE)** que tenta reconstruir **partes informativas** das observações dos outros agentes usando apenas a observação própria. ([arXiv][2])
   * Para evitar redundância, o paper introduz os **Agent Modelling (AM) filters**: vetores de pesos (w^i\in[0,1]^d) que dizem quanto cada feature dos outros agentes é relevante para o agente (i).

2. **Exploração intrínseca “adversarial” no espaço de crenças**
   Dado o embedding de crença (z_t^i), o método constrói uma **recompensa intrínseca count-based**:

   * Usa **SimHash** para projetar (z_t^i) num código discreto, conta visitas desse código e define um bônus (r^{\text{int},i}_t \propto 1/\sqrt{N_i(\text{hash}(z_t^i))}). ([arXiv][2])
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
   * AM filters (w^i) e suas redes.
   * *Replay buffer* compartilhado.

2. **Coleta de dados**
   Para cada passo de tempo:

   * Cada agente recebe observação local (o_t^i).
   * Amostra crença (z_t^i \sim q_\phi^i(z\mid\tau_t^i)).
   * Amostra ação (a_t^i \sim \pi_\theta^i(a\mid o_t^i, z_t^i)).
   * Executa ações conjuntas, recebe recompensa global (r_t) e novas observações.

3. **Cálculo de recompensa intrínseca**

   * Para cada agente, calcula hash de (z_t^i) via SimHash, atualiza contagem e computa
     [
     r^{\text{int},i}_t = \frac{\beta}{\sqrt{N_i(\text{hash}(z_t^i))}}.
     ]
   * Define a recompensa total usada no update de política como
     (\tilde r_t^i = r_t + r^{\text{int},i}_t).

4. **Armazenamento**

   * Armazena transições ((o_t, a_t, \tilde r_t, o_{t+1}, \ldots)) no *buffer*.

5. **Atualização do state modelling (VAE + AM filters)**
   Periodicamente, para cada agente:

   * Minimiza a **loss de reconstrução filtrada**
     [
     \mathcal{L}*{\text{rec}}^i
     = \mathbb{E}*{q_\phi^i(z\mid\tau)}
     \big| (o^{-i} \odot w^i) - \hat o^{-i}(z)\big|^2,
     ]
     onde (o^{-i}) são observações dos outros agentes e (\odot) é produto elemento a elemento. ([arXiv][2])
   * Adiciona:

     * termo **KL** para o VAE: (\mathrm{KL}\big(q_\phi^i(z\mid\tau),|,\mathcal{N}(0,I)\big));
     * regularização L2 nos filtros (w^i) para evitar colapso em zero.
   * A combinação desses termos define a loss de **state modelling** (\mathcal{L}_{\text{SM}}^i).

6. **Atualização dos críticos e da política**

   * Mantém dois críticos:

     * um crítico “normal” (V_\psi(s));
     * um crítico “com crenças” (V_\omega(s, z^i)) que vê o estado filtrado na perspectiva do agente. ([arXiv][2])
   * Minimiza erros TD correspondentes (loss de valor).
   * Atualiza o ator com gradiente de política usando estados estendidos ((o^i, z^i)).
   * A loss total por agente combina:
     [
     \mathcal{L}^i
     = \mathcal{L}_{\text{SM}}^i

     * \lambda_V \mathcal{L}_{\text{críticos}}^i

     - \lambda_\pi J_{\text{policy}}^i.
       ]

7. **Loop até o fim do treinamento**, com updates periódicos de *target networks* e dos encoders–decoders para estabilizar o cálculo dos bônus intrínsecos.

Matematicamente, o paper mostra que o problema de **state modelling** é formulado de forma que o valor ótimo com beliefs (z) é equivalente ao valor ótimo do Dec-POMDP original (Proposição 2.1), ou seja, introduzir (z) **não restringe** o conjunto de políticas ótimas possíveis – ele só reparametriza o problema de forma mais informativa. ([arXiv][2])

---

### Nossa contribuição: SMPE + HPO evolucionário em AgileRL

A implementação deste repositório parte do **código oficial do SMPE/SMPE²** ([GitHub][1]) e o adapta para o ecossistema da biblioteca **[AgileRL](https://github.com/AgileRL/AgileRL)**, que é focada em **hiperparameter optimization (HPO) evolucionário** para RL. ([GitHub][3])

A principal ideia é:

* **Encapsular os componentes centrais do SMPE** (encoder–decoder de state modelling, AM filters, críticos adicionais, cálculo de bônus intrínseco) como um **`EvolvableModule`** do AgileRL. ([docs.agilerl.com][4])
* Permitir que a infraestrutura de **HPO evolucionário** do AgileRL faça *mutation* e *selection* não só de hiperparâmetros clássicos (learning rates, coeficientes (\lambda), dimensão do embedding (z), peso da recompensa intrínseca (\beta) etc.), mas também de **submódulos da arquitetura SMPE**.

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

Na prática, isso significa que:

* o **ajuste fino dos hiperparâmetros sensíveis** do SMPE (e.g., pesos dos termos de loss, arquiteturas do VAE, intensidade de exploração intrínseca) é **automatizado**;
* o algoritmo pode ser reusado em novos cenários (por exemplo, diferentes tarefas da MPE, LBF, RWARE ou outros ambientes PettingZoo) **sem precisar redesenhar o tuning** do zero;
* ganha-se um pipeline mais próximo de uso real: **um único experimento com população evolutiva encontra tanto a política SMPE quanto uma configuração razoável de hiperparâmetros.**

[1]: https://github.com/ddaedalus/smpe "GitHub - ddaedalus/smpe: [ICML 2025] Official Code of SMPE: \"Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration\""
[2]: https://arxiv.org/html/2505.05262v1 "Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration"
[3]: https://github.com/AgileRL/AgileRL?utm_source=chatgpt.com "AgileRL/AgileRL"
[4]: https://docs.agilerl.com/en/latest/api/modules/base.html?utm_source=chatgpt.com "EvolvableModule"
[5]: https://docs.agilerl.com/en/latest/evo_hyperparam_opt/index.html?utm_source=chatgpt.com "Evolutionary Hyperparameter Optimization"


