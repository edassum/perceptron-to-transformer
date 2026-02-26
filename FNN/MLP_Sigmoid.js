/**
 * A simple, node-based Multi-Layer Perceptron (MLP) to solve the non-linar problem.
 */

// --- Helper Activation Functions ---
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(y) {
    // y is the output of the sigmoid function
    return y * (1 - y);
}

/**
 * Represents a single neuron in the network.
 */
class Neuron {
    constructor(nInputs) {
        // Initialize weights randomly for each input, and a random bias.
        this.weights = [];
        for (let i = 0; i < nInputs; i++) {
            this.weights.push(Math.random() * 2 - 1);
        }
        this.bias = Math.random() * 2 - 1;

        // State variables that change during training
        this.output = 0; // Last calculated output
        this.delta = 0;  // The error term (error * derivative of activation)
    }

    /**
     * Calculates the output of this neuron for a given set of inputs.
     * @param {number[]} inputs - The inputs to the neuron.
     * @returns {number} The calculated output.
     */
    activate(inputs) {
        let activation = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            activation += this.weights[i] * inputs[i];
        }
        this.output = sigmoid(activation);
        return this.output;
    }
}

/**
 * Represents a layer of neurons.
 */
class Layer {
    constructor(nNeurons, nInputsPerNeuron) {
        this.neurons = [];
        for (let i = 0; i < nNeurons; i++) {
            this.neurons.push(new Neuron(nInputsPerNeuron));
        }
    }

    /**
     * Feeds an array of inputs through all neurons in the layer and returns output.
     * @param {number[]} inputs - The inputs for the layer.
     * @returns {number[]} An array of outputs from each neuron.
     */
    feedForward(inputs) {
        const outputs = [];
        for (let i = 0; i < this.neurons.length; i++) {
            outputs.push(this.neurons[i].activate(inputs));
        }
        return outputs;
    }
}

/**
 * The main network class, composed of layers.
 */
class MultiLayerPerceptron {
    constructor(nInputs, nHidden, nOutputs) {
        this.hiddenLayer = new Layer(nHidden, nInputs);
        this.outputLayer = new Layer(nOutputs, nHidden);
    }

    // --- Feedforward (Prediction) ---
    predict(inputs) {
        const hiddenOutputs = this.hiddenLayer.feedForward(inputs);
        const finalOutputs = this.outputLayer.feedForward(hiddenOutputs);
        return finalOutputs;
    }

    /**
     * Performs one training step (forward pass, backpropagation, and weight update).
     * @param {number[]} inputs - A single training input sample.
     * @param {number[]} targets - The corresponding target outputs.
     * @param {number} learningRate - The learning rate.
     */
    train(inputs, targets, learningRate) {
        // 1. Forward Pass (get outputs from all neurons)
        const hiddenOutputs = this.hiddenLayer.feedForward(inputs);
        this.outputLayer.feedForward(hiddenOutputs);

        // 2. Backward Pass (Backpropagate error and calculate deltas)

        // Calculate deltas for the output layer neurons
        for (let i = 0; i < this.outputLayer.neurons.length; i++) {
            const neuron = this.outputLayer.neurons[i];
            const error = targets[i] - neuron.output;
            neuron.delta = error * sigmoidDerivative(neuron.output);
        }

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

    // --- Training the Network using Backpropagation ---
    fit(X, y, epochs = 10000, learningRate = 0.1) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < X.length; i++) {
                this.train(X[i], y[i], learningRate); // Train on one sample
            }
        }
    }
}

// --- Create multi-layer perceptorn network ---
const nn = new MultiLayerPerceptron(2, 4, 1); // 2 inputs, 4 hidden neurons, 1 output neuron

const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y = [[1], [0], [0], [1]]; // XNOR

console.log("Training the MLP to solve XNOR...");
nn.fit(X, y, 10000, 0.1);

console.log("\n--- Testing after training ---");
for (let i = 0; i < X.length; i++) {
    const input = X[i];
    const prediction = nn.predict(input);
    console.log(`Input: [${input.join(', ')}], Target: ${y[i][0]}, Prediction: ${prediction[0].toFixed(4)} -> Rounded: ${Math.round(prediction[0])}`);
}
