using MathNet.Numerics.LinearAlgebra;

namespace server_app.neuralNetwork
{
    // neural network
    public class @evaluate
    {
        public static readonly int[] layerSizes = [784, 144, 72, 26];
        public static readonly int layerCount = layerSizes.Length;

        public double[][] neuronValues = new double[layerCount][];
        public double[][] activatedValues = new double[layerCount][];

        public double[][,] weights = new double[layerCount - 1][,];
        public double[][] biases = new double[layerCount - 1][];

        public int? result;
        public evaluate(double[] input)
        {
            // initialise input layer
            neuronValues[0] = input;
            activatedValues[0] = input;

            // load weights and biases
            weights = data.loadWeights();
            biases = data.loadBiases();

            evaluateNetwork();
        }
        public evaluate(double[] input, double[][,] weights, double[][] biases)
        {
            // initialises input layer
            neuronValues[0] = input;
            activatedValues[0] = input;

            // assign loaded weights and biases
            this.weights = weights;
            this.biases = biases;

            evaluateNetwork();
        }
        public void evaluateNetwork()
        {
            // for each layer excluding the input layer
            for (int layer = 1; layer < layerCount; layer++)
            {
                // build matrices
                Vector<double> neuronsMatrix = Vector<double>.Build.DenseOfArray(activatedValues[layer - 1]);

                Matrix<double> weightsMatrix = Matrix<double>.Build.DenseOfArray(weights[layer - 1]);
                Vector<double> biasesMatrix = Vector<double>.Build.DenseOfArray(biases[layer - 1]);

                // calculate neuron values
                neuronValues[layer] = (neuronsMatrix * weightsMatrix + biasesMatrix).ToArray();

                // calculate activated values
                activatedValues[layer] = new double[layerSizes[layer]];
                if (layer == layerCount - 1)
                {
                    activatedValues[layer] = softmax(neuronValues[layer]);
                }
                else
                {
                    for (int i = 0; i < neuronValues[layer].Length; i++)
                    {
                        activatedValues[layer][i] = sigmoid(neuronValues[layer][i]);
                    }
                }
                
            }
            // output letter as integer from 0-25
            result = activatedValues[layerCount - 1].ToList().IndexOf(activatedValues[layerCount - 1].Max());
        }
        private static double sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        private static double[] softmax(double[] input)
        {
            double[] output = new double[input.Length];
            double sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Math.Exp(input[i]);
                sum += output[i];
            }
            for (int i = 0; i < output.Length; i++)
            {
                output[i] /= sum;
            }
            return output;
        }
    }
}
