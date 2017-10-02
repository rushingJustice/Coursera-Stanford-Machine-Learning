function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Cost Function
g_thetax = sigmoid(X*theta);
% initial J and grad
J = (-1/m)*(y'*log(g_thetax) + (1 - y')*log(1 - g_thetax)); 
grad = (1/m) * X'*(g_thetax - y);

% J_j and grad_j and combine
theta(1) = 0; % set theta_0 to zero for regularization
J = J + lambda/(2*m) *sum(theta.^2); % full cost with reg
grad = grad + lambda/m*theta;

% =============================================================

end
