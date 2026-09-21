# Hierarchical Reasoning Model (HRM) on Sudoku Puzzles

A Minor Project focused on implementing and evaluating the **Hierarchical Reasoning Model (HRM)** for solving challenging 9×9 Sudoku puzzles. The project studies whether hierarchical recurrent reasoning can achieve strong computational depth and effective backtracking while learning from a very small number of training examples.

## 👥 Project Team

* **Hamza Ehsan** — 23AIBEA627
* **Md. Dilnawaz Hussain** — 22AIB470
* **Supervisor:** Miss Ayesha Khan
* **Project:** Minor Project - 1

---

## 📌 Project Overview

The **Hierarchical Reasoning Model (HRM)** is a recurrent neural network architecture inspired by hierarchical and multi-timescale processing. Instead of relying on explicit Chain-of-Thought (CoT) supervision, HRM uses two interdependent reasoning modules:

* **High-Level (H) Module** — performs higher-level planning and maintains global context.
* **Low-Level (L) Module** — performs faster, detailed computations and local constraint checking.

The project applies HRM to **Sudoku**, a constraint-satisfaction problem that requires search, reasoning, and backtracking.

The Sudoku input is represented as an 81-token sequence corresponding to the 9×9 grid. The model predicts the solved Sudoku grid while iteratively refining its intermediate states.

---

## 🎯 Objectives

1. Study the Hierarchical Reasoning Model and its recurrent reasoning mechanism.
2. Train HRM using only **1,000 Sudoku training examples** with data augmentation.
3. Evaluate HRM on unseen Sudoku puzzles.
4. Compare HRM with:

   * Recurrent Transformer
   * Recurrent Relational Network (RRN)
   * SATNet
5. Analyze intermediate reasoning timesteps and backtracking behaviour.
6. Investigate data-efficient reasoning for difficult constraint-based problems.

---

## 🧩 Problem Statement

Develop a reasoning architecture capable of handling tasks that require extensive search and backtracking while using a minimal amount of training data.

Sudoku is used as the test problem because every prediction must satisfy constraints across:

* Rows
* Columns
* 3×3 boxes

---

## 📊 Dataset

The project uses the **Sudoku-Extreme** dataset.

### Dataset Characteristics

* Sudoku puzzles are represented by a `question` field containing an incomplete puzzle.
* The `answer` field contains the corresponding solved puzzle.
* A rating/backtracking-related field is also associated with the dataset.
* The project uses **1,000 original training examples** and applies augmentation to increase the effective training diversity.

### Input Representation

Each Sudoku board contains:

```text
9 × 9 = 81 cells
```

The puzzle is flattened into an 81-token sequence.

Token mapping:

```text
'.' → 0
'1' → 1
'2' → 2
...
'9' → 9
```

The integer sequence is converted into a tensor and passed through an embedding layer.

---

## 🔄 Data Augmentation

Four rule-preserving transformations are used:

### 1. Digit Permutation

Randomly permutes digits 1–9 while keeping blank cells unchanged.

### 2. Transposition

With 50% probability, rows and columns are swapped.

### 3. Band Shuffling

The three groups of rows are permuted, and rows within each band can also be shuffled.

### 4. Stack Shuffling

The three groups of columns are permuted, and columns within each stack can also be shuffled.

These transformations preserve the underlying Sudoku constraints while producing diverse training examples.

---

## 🏗️ HRM Architecture

The overall pipeline is:

```text
Sudoku Puzzle
      ↓
Tokenization
      ↓
Integer Index Mapping
      ↓
Embedding Layer
      ↓
┌───────────────────────────────┐
│     Hierarchical Reasoning    │
│                               │
│   High-Level (H) Module       │
│             ↕                 │
│   Low-Level (L) Module        │
│                               │
└───────────────────────────────┘
      ↓
    Q-Head
      ↓
Halt / Continue Decision
      ↓
Output Head
      ↓
Softmax Probabilities
      ↓
Solved Sudoku
```

### Main Components

#### Embedding Layer

Converts discrete Sudoku tokens into vector representations.

#### High-Level Module (H)

An encoder-only Transformer block operating at a slower reasoning timescale. It integrates information from the lower-level computations and supports higher-level planning.

#### Low-Level Module (L)

An encoder-only Transformer block that performs faster, detailed reasoning and local constraint checking.

#### Q-Head

Uses a Q-learning based decision mechanism to determine whether the model should:

```text
HALT     → stop reasoning
CONTINUE → perform another reasoning step
```

This forms the basis of **Adaptive Computation Time (ACT)**.

#### Output Layer

Transforms hidden states into logits, which are converted into probabilities using Softmax.

---

## 🧠 Training & Optimization

The project uses several techniques to make recurrent reasoning computationally practical.

### One-Step Gradient Approximation

Instead of backpropagating through the complete unrolled recurrent computation, the gradient is approximated through the final step of each reasoning cycle.

### Deep Supervision

Loss feedback is provided at the end of reasoning segments instead of evaluating only the final prediction.

### Adaptive Computation Time

The Q-Head dynamically decides whether the model should continue reasoning or halt.

### Recurrent Reasoning

The reasoning process can be viewed as:

```text
Input
  ↓
L-Module performs several detailed steps
  ↓
H-Module updates the high-level state
  ↓
Reasoning cycle repeats
  ↓
Model refines predictions / backtracks
  ↓
Q-Head decides HALT or CONTINUE
  ↓
Final Sudoku
```

---

## 📏 Evaluation Metrics

### 1. Cell-Wise Accuracy

Accuracy is calculated over the cells that need to be predicted.

```text
Accuracy =
Correctly Predicted Blank Cells
-------------------------------- × 100
Total Blank Cells
```

### 2. Average Backtracking

Backtracking measures how often the model changes its predictions while attempting to satisfy Sudoku constraints.

A lower average backtracking value indicates fewer prediction changes during the reasoning process.

---

## 🧪 Models Compared

The project compares four approaches:

| Model                     | Main Idea                                                                  |
| ------------------------- | -------------------------------------------------------------------------- |
| **HRM**                   | Hierarchical recurrent reasoning with H/L modules and adaptive computation |
| **Recurrent Transformer** | Recurrent Transformer-based reasoning using self-attention                 |
| **RRN**                   | Graph-based recurrent message passing between related Sudoku cells         |
| **SATNet**                | Differentiable satisfiability-based reasoning                              |

---

## 📈 Results

The final comparison table presented in the project presentation reports:

| Model                 | Cell Accuracy | Avg. Backtracking | Final Loss |
| --------------------- | ------------: | ----------------: | ---------: |
| **HRM**               |     **71.4%** |          **1.83** | **0.1649** |
| Recurrent Transformer |        52.38% |              4.72 |     0.2039 |
| RRN                   |        45.72% |              7.41 |     1.1006 |
| SATNet                |        36.22% |             43.81 |     1.3975 |

---

## 🔍 Key Observations

* HRM achieves the highest cell-wise accuracy reported in the project's final comparison table.
* HRM shows the lowest average backtracking among the compared models.
* The hierarchical H/L structure provides separate levels of reasoning.
* Adaptive Computation Time allows the model to decide when to continue or stop computation.
* Intermediate timestep visualizations provide insight into how predictions evolve during reasoning.
* The project investigates reasoning under a low-data training setting rather than relying on massive pretraining datasets.

---

## 📁 Suggested Repository Structure

```text
HRM-Sudoku/
│
├── README.md
├── data/
│   └── sudoku-extreme/
│
├── models/
│   ├── hrm.py
│   ├── recurrent_transformer.py
│   ├── rrn.py
│   └── satnet.py
│
├── training/
│   ├── train_hrm.py
│   └── evaluation.py
│
├── results/
│   ├── hrm_act_timesteps.png
│   ├── hrm_results/
│   └── visualizations/
│
├── notebooks/
│   └── experiments.ipynb
│
└── requirements.txt
```

> The structure above is a suggested organization. Adapt the filenames to the actual files present in the repository.

---

## ⚙️ Technologies & Concepts

* Python
* PyTorch
* Deep Learning
* Recurrent Neural Networks
* Transformer Architecture
* Reinforcement Learning
* Q-Learning
* Adaptive Computation Time (ACT)
* Sudoku Constraint Satisfaction
* Data Augmentation
* Tensor Embeddings
* Backtracking
* Multi-class Classification

---

## 🚀 How the System Works

1. Load a Sudoku puzzle.
2. Convert the puzzle into an 81-token sequence.
3. Map digits and blank cells to integer IDs.
4. Convert the IDs into learnable embeddings.
5. Process the embeddings through the recurrent H/L reasoning modules.
6. Repeatedly refine the Sudoku predictions.
7. Use the Q-Head to decide whether to halt or continue.
8. Convert the final logits into Sudoku digit predictions.
9. Evaluate the prediction using cell-wise accuracy and average backtracking.
10. Visualize intermediate reasoning timesteps.

---

## 📚 References

1. Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., & Yadkori, Y. A. (2025). **Hierarchical Reasoning Model**. arXiv:2506.21734.
2. Palm, R., Paquet, U., & Winther, O. (2018). **Recurrent Relational Networks**. Advances in Neural Information Processing Systems, 31.
3. Wang, P. W., Donti, P., Wilder, B., & Kolter, Z. (2019). **SATNet: Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver**. International Conference on Machine Learning, PMLR.
4. Yang, Z., Ishay, A., & Lee, J. (2023). **Learning to Solve Constraint Satisfaction Problems with Recurrent Transformer**. arXiv:2307.04895.

---

## 📄 Project Documentation

The repository is accompanied by the project report and presentation covering:

* Introduction and motivation
* Literature review
* System design
* Dataset and augmentation
* HRM architecture
* Training methodology
* Evaluation metrics
* Comparative results
* Intermediate reasoning visualizations
* Conclusions and references

---

## 👨‍💻 Authors

**Hamza Ehsan**
Faculty No.: 23AIBEA627

**Md. Dilnawaz Hussain**
Faculty No.: 22AIB470

**Supervisor:** Miss Ayesha Khan

**Minor Project - 1**
