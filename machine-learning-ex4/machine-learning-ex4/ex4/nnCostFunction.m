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

% Add intercept to X
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


%% Forward Prop for Cost Function
% Calculate h_k [m x k] (3 layer network)
z2 = X*Theta1';
a2 = sigmoid(z2); a2 = [ones(m, 1) a2];

z3 = a2*Theta2'; %[m x K]
a3 = sigmoid(z3);
    
%% Cost Function
for c = 1:num_labels
    J =  J + ((y==c)'*log(a3(:,c)) + (1 - (y==c)')*(log(1 - a3(:,c)))); 
end   
    
% Regularization
tempTheta1 = Theta1; tempTheta2 = Theta2;
tempTheta1(:,1) = []; tempTheta2(:,1) = [];

reg = lambda/(2*m) * (sum((sum(tempTheta1.^2))) + ...
    sum((sum(tempTheta2.^2))));

% Calculate cost with reg
J = -1/m * J + reg;
%
%% Part 2: Implement the backpropagation algorithm to compute the gradients
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

% Calculate binary y_k matrix (K outputs for each example m]
y_k = [1:num_labels] == y; %[m x K]
tempTheta1 = Theta1; tempTheta1(:,1) = [];
tempTheta2 = Theta2; tempTheta2(:,1) = [];


% Step 1 - Forward Prop
a1 = X; z2 = a1*Theta1'; % [1x25] [m x 25]
a2 = sigmoid(z2); a2 = [ones(m, 1) a2]; % [1x26] [m x 26]
z3 = a2*Theta2'; a3 = sigmoid(z3); % [1x10] [m x K]

% Step 2 - Compute error of Output Layer
del3 = (a3 - y_k); % [1x10] [m x K]

% Step 3 - Compute error of Hidden Layer
del2 = del3 * tempTheta2 .* sigmoidGradient(z2); % [1x25] [m x 25]

% Step 4 - Accumulate gradient (for example t)
Theta1_grad = Theta1_grad + del2'*a1; % [25x401] [25 x n+1]
Theta2_grad = Theta2_grad + del3'*a2; % [10x26] [K x 26]

% Step 5 - Divide sum of accumulated gradients by total training examples
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
tempTheta1 = [zeros(size(Theta1,1),1) tempTheta1];
tempTheta2 = [zeros(size(Theta2,1),1) tempTheta2];
Theta1_grad = Theta1_grad + lambda/m * tempTheta1;
Theta2_grad = Theta2_grad + lambda/m * tempTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
