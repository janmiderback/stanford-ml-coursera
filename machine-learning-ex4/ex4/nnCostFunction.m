function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Forward propagation to get h. Expanding with column of ones as needed.
h1 = sigmoid([ones(m, 1) X] * Theta1');
h = sigmoid([ones(m, 1) h1] * Theta2');

%% Reformat y to contain vectors instead.
y1 = zeros(m, num_labels);
for r = 1:m
    y1(r, y(r)) = 1;
end

%% Compute cost. Need to sum over both columns and rows.
J = sum(sum(-y1 .* log(h) - (1 - y1) .* log(1 - h))) / m;

%% Regularize. Omit first column of weight matrices.
T1 = Theta1;
T1(:, 1) = [];
T1 = T1 .^ 2;

T2 = Theta2;
T2(:, 1) = [];
T2 = T2 .^ 2;

S1 = sum(sum(T1));
S2 = sum(sum(T2));
J = J + (S1 + S2) * lambda / (2 * m);

%% Compute gradient of cost function using backpropagation.
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% Step 1 - Feedforward pass.
a1 = [ones(m, 1) X]';
z2 = Theta1 * a1;
a2 = [ones(1, m); sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% Step 2 - Calculate deltas for output units.
d3 = a3 - y1';
    
% Step 3 - Calculate deltas for hidden layers (only one).
T2 = Theta2;
T2(:, 1) = [];
d2 = (T2' * d3) .* sigmoidGradient(z2);
    
% Step 4 - Accumulate the gradients.
Delta1 = Delta1 + d2 * a1';
Delta2 = Delta2 + d3 * a2';

% Step 5 - Obtain final gradients.
Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

% Regularization terms.
RTheta1 = Theta1;
RTheta1(:, 1) = 0;  % Set col 1 to 0 (bias terms).
RTheta2 = Theta2;
RTheta2(:, 1) = 0;

Theta1_grad = Theta1_grad + (lambda / m) * RTheta1;
Theta2_grad = Theta2_grad + (lambda / m) * RTheta2;

%%
% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
