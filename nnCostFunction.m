function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
h = size(Theta1, 1);
r = num_labels;
                      
                               
%eye_matrix = eye(num_labels);
%y_matrix = eye_matrix(y,:);
y_matrix = bsxfun(@eq, y, 1:num_labels);

% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
a1 = [ones(m,1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

Cost = sum(sum(  -y_matrix.*log(a3) - (1 - y_matrix).*log(1 - a3)  ));
Reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = (1/m)*Cost + Reg;
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.
d3 = a3 - y_matrix;

d2 = d3*Theta2(:,2:end) .* sigmoidGradient(z2);

Delta1 = d2'* a1;
Delta2 = d3' * a2;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = (1/m)*Delta1 + (lambda/m)*Theta1;
Theta2_grad = (1/m)*Delta2 + (lambda/m)*Theta2;



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
