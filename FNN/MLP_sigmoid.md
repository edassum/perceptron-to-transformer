### Traing Steps

The training process uses **Stochastic Gradient Descent (SGD)**. Below are the calculus-based formulas applied in the `train()` method to minimize the error using the Chain Rule.

| Step | Process | Mathematical Formula |
| :--- | :--- | :--- |
| **1** | **Output Layer Delta ($\delta_o$)** | $\delta_o = (T - O) \cdot O(1 - O)$ |
| **2** | **Hidden Layer Delta ($\delta_h$)** | $\delta_h = (\sum w_o \delta_o) \cdot H(1 - H)$ |
| **3** | **Weight Update ($w$)** | $w = w + (\eta \cdot \delta \cdot \text{input})$ |
| **4** | **Bias Update ($b$)** | $b = b + (\eta \cdot \delta)$ |

---

### 🔍 Variable Definitions

* **$T$**: Target value (the ground truth).
* **$O$ / $H$**: The actual Output of the neuron (calculated via Sigmoid).
* **$O(1 - O)$**: The derivative of the Sigmoid activation function.
* **$\eta$ (Eta)**: The **Learning Rate**. This determines the size of the step we take toward the minimum error.
* **$\delta$ (Delta)**: The error gradient (the "responsibility" a neuron takes for the total error).
