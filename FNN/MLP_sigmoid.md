### Training Steps

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

**1** **Output Layer Delta ($\delta_o$)** | $\delta_o = (T - O) \cdot O(1 - O)$ 
```javascript

 for (let i = 0; i < this.outputLayer.neurons.length; i++) {
            const neuron = this.outputLayer.neurons[i];
            const error = targets[i] - neuron.output;
            neuron.delta = error * sigmoidDerivative(neuron.output);
        }
```
**2** | **Hidden Layer Delta ($\delta_h$)** | $\delta_h = (\sum w_o \delta_o) \cdot H(1 - H)$ 
```javascript
        // Calculate deltas for the hidden layer neurons
        for (let j = 0; j < this.hiddenLayer.neurons.length; j++) {
            const hiddenNeuron = this.hiddenLayer.neurons[j];
            let error = 0.0;
            // Sum the contributions of this hidden neuron's output to the errors of all output neurons
            for (let k = 0; k < this.outputLayer.neurons.length; k++) {
                const outputNeuron = this.outputLayer.neurons[k];
                error += outputNeuron.weights[j] * outputNeuron.delta;
            }
            hiddenNeuron.delta = error * sigmoidDerivative(hiddenNeuron.output);
        }
```
**3O** | **Weight Update Output Layer ($w$)** | $w = w + (\eta \cdot \delta \cdot \text{input})$
**4O** | **Bias Update Output Layer ($b$)** | $b = b + (\eta \cdot \delta)$
```javascript
        // 3. Update all weights and biases in the network

        // Update output layer weights
        for (let i = 0; i < this.outputLayer.neurons.length; i++) {
            const neuron = this.outputLayer.neurons[i];
            for (let j = 0; j < neuron.weights.length; j++) {
                // hiddenOutputs[j] is the input to this output neuron
                neuron.weights[j] += learningRate * neuron.delta * hiddenOutputs[j];
            }
            neuron.bias += learningRate * neuron.delta;
        }
```
**3H** | **Weight Update Hidden Layer ($w$)** | $w = w + (\eta \cdot \delta \cdot \text{input})$
**4H** | **Bias Update Hidden Layer ( $b$)** | $b = b + (\eta \cdot \delta)$
```javascript
        // Update hidden layer weights
        for (let i = 0; i < this.hiddenLayer.neurons.length; i++) {
            const neuron = this.hiddenLayer.neurons[i];
            for (let j = 0; j < neuron.weights.length; j++) {
                // inputs[j] is the input to this hidden neuron
                neuron.weights[j] += learningRate * neuron.delta * inputs[j];
            }
            neuron.bias += learningRate * neuron.delta;
        }
    }
    ```
