# Mridangam_Reinforcement_Learning
Algorithmic Percussion Generation of Carnatic Mridangam Mora Patterns in Adi Tala Using Temporal Difference Reinforcement Learning
This proposal outlines a formal methodology for modeling and generating mathematically constrained rhythmic cadences (Moras) in South Indian Carnatic percussion via Model-Free Temporal Difference Reinforcement Learning (TD RL). Unlike Western music composition frameworks focused primarily on polyphonic harmonic progressions, Carnatic percussion—exemplified by the double-headed drum, Mridangam—is strictly governed by metric time cycles (Talas), speed subdivisions (Gathis), and rhythmic resolutions (Moras).

# Algorithmic Percussion Generation of Carnatic Mridangam Mora Patterns in Adi Tala Using Temporal Difference Reinforcement Learning

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Reinforcement Learning framework that formulates South Indian Carnatic percussion cadence generation (*Mridangam Moras*) as a discrete Markov Decision Process (MDP). The project implements a model-free Tabular Temporal Difference (TD) Q-Learning agent operating in a custom OpenAI Gymnasium environment (`MridangamMoraEnv` / `MridangamKorvaiEnv`) to achieve 100% zero-violation cadence resolution on *Samam* (beat 0 of the succeeding metric cycle).

---

## 📌 Project Overview

In South Indian Carnatic classical music, percussive performance on the **Mridangam** is strictly governed by metric time cycles (*Talas*), subdivision rates (*Gathis*), and resolution cadences (*Moras*). A classical *Mora* features a body section (*Purvanga*) followed by a terminal resolution section (*Uttaranga*) consisting of three identical phrase repetitions ($P$) separated by two equal silent rest gaps ($G$), conforming to the structural form:

$$P_1 + G_1 + P_2 + G_2 + P_3$$

The primary metric constraint requires the terminal stroke of the *Mora* to resolve precisely on **Samam** (beat 0 of the next cycle) with zero metric error over a 16-decision step horizon ($4 + 2 + 4 + 2 + 4 = 16$ beats under *Adi Tala* in *Chatushra Gathi*).

Traditional rule-based systems struggle with dynamic constraint flexibility, while supervised models require static labelled audio datasets. This project utilizes model-free **TD Q-Learning** to learn optimal rhythmic action selection purely through trial-and-error interaction with a custom simulator.

---

## 📐 Markov Decision Process (MDP) Formulation

The task is formally modeled as a 5-tuple discrete MDP $(S, A, P_{\text{trans}}, R, \gamma)$:


                 +---------------------------------------+
                 |              ENVIRONMENT              |
                 |  - Custom Gymnasium Class (Python)    |
                 |  - Matra Tracker (16 -> 0)            |
                 |  - Tala Rule Verification             |
                 +---------------------------------------+
                              ^             |
               Reward R(s,a) |             | State S_t
                & Next State |             | [pos, phase,
                             |             |  p_count, g_count]
                             |             v
                 +---------------------------------------+
                 |                 AGENT                 |
                 |        (Tabular TD Q-Learning)        |
                 +---------------------------------------+
                             |             ^
                             |  Action A_t |
                             +-------------+
                   (Select Sollukattu / Trigger Mora)

### State Space ($S$)
A 4-dimensional discrete integer tuple:
$$s-t = \big( C-{\text{rem}}, P-{\text{phase}}, \text{phrase\-count}, \text{gap\-count} \big)$$
- **$C_{\text{rem}}$**: Remaining decision steps in the metric cycle ($16 \to 0$).
- **$P_{\text{phase}}$**: Composition phase ($0 = \text{Purvanga}, 1 = \text{Uttaranga}$).
- **$\text{phrase\_count}$**: Accumulated steps within the active stroke phrase ($0 \dots 4$).
- **$\text{gap\_count}$**: Accumulated steps within the active rest gap ($0 \dots 2$).

### Action Space ($A$)
Five discrete symbolic choices ($a-t \in \{0, 1, 2, 3, 4\}$):
- **Active Percussive Strokes** ($0 \dots 3$): *Tha, Dhi, Thom, Nam* (Sollukattus).
- **Silent Rest / Pause** ($4$): *Karva* (pause interval).

### Reward Shaping ($R$)
$$R(s-t, a-t, s-{t+1}) = \begin{cases} +1.0 & \text{if } a-t \in \text{Strokes during Phrase phase} \\ +1.0 & \text{if } a-t = \text{Karva during Gap phase} \\ -2.0 & \text{if illegal stroke/rest placement (Rule Violation)} \\ +20.0 & \text{if } C-{\text{rem}} = 0 \text{ (Exact landing on } \mathbf{Samam}\text{)} \\ -10.0 & \text{if invalid cycle termination / overshooting} \end{cases}$$

---

## ⚡ Algorithm: Temporal Difference Q-Learning

The agent uses off-policy Tabular Q-Learning to update state-action values step-by-step via bootstrapping:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \max_{a \in A} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

- **Learning Rate ($\alpha$)**: $0.10$
- **Discount Factor ($\gamma$)**: $0.95$
- **Exploration Policy ($\epsilon$-greedy)**: Exponential decay from $\epsilon_0 = 1.00$ down to $\epsilon_{\text{min}} = 0.01$ with a decay rate of $0.995$ per episode over $1,000$ training episodes.

---

## 📊 Experimental Results

Tested in deterministic greedy evaluation rollouts ($\epsilon = 0.0$) against an untrained random baseline:

| Metric | Random Baseline Agent | Trained TD Q-Learning Agent | Optimal Direction |
| :--- | :---: | :---: | :---: |
| **Cumulative Return ($G_t$)** | $-14.67$ | **$+30.00$** | Higher ($\to +30.0$) |
| **Rule Violation Count** | $7.33$ | **$0.00$** | Lower ($\to 0$) |
| **Cycle Duration Error** | Non-zero / Variable | **$0$ Beats** | Lower ($\to 0$) |
| **Samam Success Rate** | $0.0\%$ | **$100.0\%$** | Higher ($\to 100\%$) |

### Generated Sequence Output ($\epsilon = 0.0$)
$$\underbrace{\text{Tha -- Dhi -- Thom -- Nam}}_{P_1 = 4 \text{ beats}} \;\mid\; \underbrace{\text{Karva -- Karva}}_{G_1 = 2 \text{ beats}} \;\mid\; \underbrace{\text{Tha -- Dhi -- Thom -- Nam}}_{P_2 = 4 \text{ beats}} \;\mid\; \underbrace{\text{Karva -- Karva}}_{G_2 = 2 \text{ beats}} \;\mid\; \underbrace{\text{Tha -- Dhi -- Thom -- Nam}}_{P_3 = 4 \text{ beats}}$$

$$\text{Total Duration} = 4 + 2 + 4 + 2 + 4 = \mathbf{16 \text{ Beats}} \quad \implies \quad \text{Exact Landing on } \mathbf{Samam}$$

---

## 🛠️ Software Stack & Dependencies

- **Language**: Python 3.10+
- **RL Framework**: OpenAI Gymnasium (`gymnasium >= 0.29.1`)
- **Computation**: NumPy (`numpy >= 1.26.4`)
- **Visualization**: Matplotlib (`matplotlib >= 3.8.0`)

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/NitzV/Mridangam_Reinforcement_Learning.git](https://github.com/NitzV/Mridangam_Reinforcement_Learning.git)
cd Mridangam_Reinforcement_Learning

Install Dependencies
pip install gymnasium numpy matplotlib

Run the Environment & Training Script
python mridangam_rl_project.py
