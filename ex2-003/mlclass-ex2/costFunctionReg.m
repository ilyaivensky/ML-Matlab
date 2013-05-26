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

for i = 1:m
    g = sigmoid(X(i,:) * theta);
    J = J + (-y(i) * log(g)) - (1 - y(i)) * log(1 - g);
    for j = 1:size(X,2)
        grad(j) = grad(j) + (g - y(i)) * X(i,j);
    end
end

grad = grad / m;
reg = 0;
for j = 2:size(X,2)
    reg = reg + (theta(j) ^ 2);
    grad(j) = grad(j) + (lambda * theta(j)) / m;
end

reg = reg * lambda / (2 * m);
J = J / m + reg;


% ================================================================

% g = sigmoid(X * theta);
% tmp_J = -y .* log(g) - (1 - y) .* log(1 - g); 
% tmp_reg = theta .^2;
% reg = sum(tmp_reg) * lambda / (2 * m);
% J = sum(tmp_J) / m + reg;
% 
% grad = X' * (g - y);
% grad = grad / m;
% 
% temp = theta; 
% temp(1) = 0;
% grad = grad + (lambda / m) * temp;

%J
%grad

% =============================================================

end