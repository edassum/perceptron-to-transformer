# perceptron-to-transformer

## From First Principles to Modern Neural Networks

This repository documents my deep learning learning journey — starting from the **Perceptron** and gradually progressing toward modern **Transformer-based architectures**.

The focus of this project is **understanding how neural networks work internally**, not just how to use high-level frameworks.  
Every concept is approached from first principles, supported by minimal, readable implementations.

---

## 🎯 Goals

- Build a **strong conceptual foundation**
- Understand the **mathematics behind learning**
- Implement models **from scratch**
- See how modern architectures evolved from simpler ones
- Connect **theory ↔ code ↔ intuition**

---

## 🧠 Learning Philosophy

> **Learn → Implement → Experiment → Understand**

- Learn the idea
- Implement it manually
- Experiment with variations
- Understand *why* it works (or fails)

All implementations aim to be:
- Simple
- Educational
- Explicit
- Easy to modify and extend

---

## 🗂️ Intended Repository Structure

The code and notes are organized to reflect a **logical learning progression**:

```text
perceptron-to-transformer/
│
├── concepts/        # Theory, math, and intuition
│   ├── perceptron.md
│   ├── loss-functions.md
│   ├── gradient-descent.md
│   ├── backpropagation.md
│   └── activation-functions.md
│
├── fnn/             # Feedforward Neural Networks (MLP)
│   ├── single-layer/
│   ├── multi-layer/
│   └── experiments/
│
├── cnn/             # Convolutional Neural Networks
│   ├── convolution-from-scratch/
│   ├── pooling/
│   └── cnn-models/
│
├── rnn/             # Recurrent Neural Networks
│   ├── vanilla-rnn/
│   ├── lstm/
│   └── gru/
│
├── attention/       # Attention mechanisms
│   └── self-attention-from-scratch/
│
├── transformers/    # Transformer architecture
│   ├── embeddings/
│   ├── multi-head-attention/
│   └── transformer-from-scratch/
│
└── notes/           # Reflections, comparisons, and insights
