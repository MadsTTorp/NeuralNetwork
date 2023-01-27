import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNetwork: 
                        
    # Fun to initialize parameters
    def initialize_parameters(self, layers_dims):
    ''' 
    Input: A list with the network architecture 
    Output: initial weights and biases given the input architecture (bias = 0 and weight small)
    '''
        # set seed
        np.random.seed(3) 
        # Create dict for parameters
        parameters = {}
        # Retrieve length of network
        L = len(layers_dims)
        # loop through architecture
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
            assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))
       
        return parameters

    def initialize_velocity(self, parameters):
        '''
        Input: dict with parameters
        Output: dict with initial velocity
        '''
        # Retrieve number of layers in network
        L = len(parameters) // 2 
        # Create dict for velocity
        v = {}
        
        # loop through architecture
        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
            v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
            
        return v            
    
    def initialize_adam(self, parameters) :
        '''
        Input: dict with parameters
        Output: dict with initial ADAM algorithm
        '''
        # Retrieve number of layers in network
        L = len(parameters) // 2 # number of layers in the neural networks
        # Create dict for velocity
        v = {}
        # Create dict for s
        s = {}
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
            v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
            s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
            s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
                
        return v, s

    def relu(self, Z):
        '''
        Input: Z-values for some layer in network. 
        Output: Activation of input Z
        '''
        # Activate Z by relu function
        A = np.maximum(Z, 0)
        # Check shape is correct
        assert(A.shape == Z.shape)
        # Store Z value for backpropagation
        cache = Z
        return A, cache

    def sigmoid(self, Z):
        '''
        Input: Z-values for some layer in network. 
        Output: Activation of input Z
        '''
        # Activate Z by sigmoid function
        A = 1/(1+np.exp(-Z))
        # Store Z for backpropagation
        cache = Z
        return A, cache

    def linear_forward(self, A, W, b):
        '''
        Input: A: Sample points (or activations from previous layer)
               W: Weights for given layer in network
               b: Biases for given layer in network
        Output: Linear combination of sample points using W and b
        '''
        # Derive linear combination
        Z = np.dot(W,A) + b #b is being broadcastet
        # Store input for backpropagation
        cache = (A, W, b)
        return Z, cache 

    def linear_activation_forward(self, A_prev, W, b, activation):
        '''
        Input: A_prev: Activations from previous layer)
               W: Weights for given layer in network
               b: Biases for given layer in network
               activation: Which activation function to use (ReLU or Sigmoid)
        Output
        '''
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        # Save both caches for output
        cache = (linear_cache, activation_cache)

        return A, cache

    def l_model_forward(self, X, parameters):
        '''
        Describtion: Feeds forward in the neural network using activation in each layer
        Input: X: Sample points
               Parameters: parameters of network
        Output: Final output value, Yhat 
        '''
        # Create list for caches
        caches = []
        # Put input of first layer equal to X
        A = X
        # Retrieve number of layers in network
        L = len(parameters) // 2                
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, 
                                                W = parameters['W'+str(l)], 
                                                b = parameters['b'+str(l)], 
                                                activation = "relu")
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID (L). Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A,
                                            W = parameters['W' + str(L)],
                                            b = parameters['b' + str(L)],
                                            activation = "sigmoid")
        caches.append(cache)
            
        return AL, caches

    def compute_cost(self, AL, Y):
         '''
        Input: AL: estimate of Y (Yhat)
               Y: true value of Y
        Output: Cost of predictions
        '''
        # Retrieve number of samples
        m = Y.shape[1]
        # Compute loss from AL and Y.
        logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)    
        cost_total =  np.sum(logprobs) 
        
        return cost_total

    def linear_backward(self, dZ, cache):
        '''
        Input: dZ: gradient of Z
               cache: List of caches from forward propagation
        Output: gradient of A, W and b for single layer
        '''
        # Put variables equal to cache
        A_prev, W, b = cache
        # Retrieve number of samples from next layer
        m = A_prev.shape[1]
        # Derive gradients for W, b and A
        dW = np.dot(dZ,A_prev.T) / m
        db = np.sum(dZ, axis = 1, keepdims = True) / m
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db

    def sigmoid_backward(self, dA, cache):
        '''
        Input: dA: gradient of A
               cache: List of caches from forward propagation
        Output: gradient if Z
        '''
        Z = cache    
        s = 1/(1+np.exp(-Z))    
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ

    def relu_backward(self, dA, cache):
        '''
        Input: dA: gradient of A
               cache: List of caches from forward propagation
        Output: gradient if Z 
        '''
        Z = cache 
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.   
        # When z <= 0, you should set dz to 0 as well    
        dZ[Z <= 0] = 0 
        assert (dZ.shape == Z.shape)  
        return dZ

    def linear_activation_backward(self, dA, cache, activation):
        '''
        Input: dA: gradient of A
               cache: List of caches from forward propagation
               activation: Which activation function to use
        Output: gradient if A, W and b given activation function for single layer
        '''
        linear_cache, activation_cache = cache
        # If activation is ReLU
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache) 
        # If activation is Sigmoid   
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache) 
        
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        '''
        Input: AL: Yhat
               cache: List of caches from forward propagation
        Output: gradients of A, W and b for entire network
        '''
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dA_prev_temp, current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp

        return grads

    def update_parameters_gd(self, parameters, grads, learning_rate):
        '''
        Input: parameters: parameters of the model
               grads: gradients of A, W and b of entire network
               learning_rate: How fast gradients should descent
        Output: updated values for parameters given gradients
        '''
        parameters = parameters.copy()
        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
        return parameters
    
    def update_parameters_momentum(self, parameters, grads, v, beta, learning_rate):
        '''
        Input: parameters: parameters of the model
               grads: gradients of A, W and b of entire network
               v: Velocity
               beta: Parameter for exponential weighted moving average procedure
               learning_rate: How fast gradients should descent
        Output: updated values for parameters given gradients using momentum 
        '''
        # Retrieve number of layers in network
        L = len(parameters) // 2 
        # Momentum update for each parameter
        for l in range(1, L + 1):
    
            v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
            v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
            
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]
            
        return parameters, v

    def update_parameters_adam(self, parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon):
        '''
        Input: parameters: parameters of the model
               grads: 
               v: 
               s:
               t: 
               beta1: 
               beta2:
               learning_rate: 
        Output: 
        '''
        L = len(parameters) // 2                 # number of layers in the neural networks
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary
        
        # Perform Adam update on all parameters
        for l in range(1, L + 1):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1,t))
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1,t))

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads["dW" + str(l)],2)
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads["db" + str(l)],2)

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2,t))
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2,t))

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

        return parameters, v, s, v_corrected, s_corrected

    def predict(self, parameters, X):
        '''
        Input: parameters: parameters of the model
               X: input data
        Output: Yhat(X)
        '''
        AL, cache = self.l_model_forward(X, parameters)
        predictions = np.round(AL)
    
        return predictions

    def random_mini_batches(self, X, Y, mini_batch_size = 64, seed = 0):
        '''
        Input: X: independent input
               Y: dependent (targt) input
               mini_batch_size: size of each batch
               seed:
        Output: random split of X and Y into batch of size mini_batch_size
        '''
        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[1]                  # number of training examples
        mini_batches = []
            
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))
        
        inc = mini_batch_size

        # Step 2 - Partition (shuffled_X, shuffled_Y).
        num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]        
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    def Fit(self, X, Y, layers_dims, optimizer, learning_rate = 1e-4, mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 5000, print_cost = True, decay = None, decay_rate = 1):
        '''
        Description: The fitting of the neural network given inputs
        Input: X: independent input
               Y: dependent (targt) input
               ...
        Output: parameters of model
        '''
        # Initialize model
        t = 0
        seed = 10
        costs = []
        m = X.shape[1]
        L = len(layers_dims)
        # Initialize parameters
        parameters = self.initialize_parameters(layers_dims)
        # Initialize optimizer
        if optimizer == "gd":
            pass
        elif optimizer == "momentum":
            v = self.initialize_velocity(parameters)
        elif optimizer == "adam":
            v, s = self.initialize_adam(parameters)
        # Loop through network num_iterations-times
        for i in range(num_epochs):

            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, mini_batch_size, seed)
            cost_total = 0

            for minibatch in minibatches:

                # Choose a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                AL, caches = self.l_model_forward(minibatch_X, parameters)

                # Compute cost and add to the total
                cost_total += self.compute_cost(AL, minibatch_Y)

                # Backward propagation
                grads = self.L_model_backward(AL, minibatch_Y, caches)

                # Update parameters
                if optimizer == "gd":
                    parameters = self.update_parameters_gd(parameters, grads, learning_rate)
                elif optimizer == "momentum":
                    parameters, v = self.update_parameters_momentum(parameters, grads, v, beta, learning_rate)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    parameters, v, s, _, _ = self.update_parameters_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
            
            cost_avg = cost_total / m
            if decay:
                learning_rate0 = learning_rate
                learning_rate = decay(learning_rate0, i, decay_rate)
            
            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
            if print_cost and i % 100 == 0:
                costs.append(cost_avg)
        
        return parameters, costs

    def graph(self, costs):
        '''
        Description: Plot cost for each epoch
        Input: costs: Values of J for each epoch iteration
        Output: graph of cost
        '''
        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

