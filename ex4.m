%% Initialization
clear ; close all; clc



% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)


% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

%load('ex4data1.mat');
X = loadMNISTImages('train-images.idx3-ubyte')';
y = loadMNISTLabels('train-labels.idx1-ubyte');
X_test = loadMNISTImages('t10k-images.idx3-ubyte')';
y_test = loadMNISTLabels('t10k-labels.idx1-ubyte');
m = size(X, 1);

% Randomize
Data = [X, y];
ordering = randperm(m);
Data = Data(ordering, :);
X = Data(:,1:input_layer_size);
y = Data(:,end);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

%displayData(X(sel, :));

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Training NN ===================
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 500);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Visualize Weights =================
fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta1(:, 2:end));



%% ================= Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);


