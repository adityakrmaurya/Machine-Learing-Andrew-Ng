function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    % m = no of training examples
    % n = no of features
    % dimension(X) = m * (n+1)
    h = X * theta;  % dimension(h) = (m * (n + 1)) * ((n+1) * 1) = m * 1
    errors_vector = h - y;  % dimension(error_vector) = m * 1
    gradient = (alpha * (X' * errors_vector)) / m;
    % the simultaneous update
    theta = theta - gradient;
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
