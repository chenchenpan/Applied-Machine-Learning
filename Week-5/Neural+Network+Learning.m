
warning('off'); addpath('../readonly/Assignment4/');
load('ex4data1.mat');
m = size(X, 1);
                          % Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

Theta2(1:10,1:10)

load('ex4weights.mat');                 % Load the weights into variables Theta1 and Theta2
nn_params = [Theta1(:) ; Theta2(:)];    % Unroll parameters 
lambda = 0;
whos

y(2000:2010,:)

% GRADED FUNCTION: nnCostFunction
function [J grad] = nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);                               % Setup some useful variables
J = 0;                                        % You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

% PART 1: FEED FORWARD PROPAGATION
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2_new = [ones(m,1) a2];
z3 = a2_new * Theta2';
a3 = sigmoid(z3);

J = (-1/m)*sum(sum(y_matrix.*log(a3)+(1-y_matrix).*log(1-a3)));

% PART 2: BACK PROPAGATION
d3 = a3 - y_matrix;
d2 = (d3 * Theta2(:,2:end)).* sigmoid(z2).* (1-sigmoid(z2));

delta1 = (d2' * a1) .* (1/m);
delta2 = (d3' * a2_new) .* (1/m);


% PART 3: WEIGHT REGULARIZATION
J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;
Theta1_grad = (lambda/m)*Theta1_reg + delta1;
Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;
Theta2_grad = (lambda/m)*Theta2_reg + delta2;

% ============================================================

grad = [Theta1_grad(:) ; Theta2_grad(:)];     % Unroll gradients

end

% Testing your nnCostFunction for the non regularized cost Function 
lambda = 1;                                   % Change lambda to 1 when checking for regularization
[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
J

% GRADED FUNCTION: sigmoidGradient
function g = sigmoidGradient(z)

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
g = sigmoid(z).*(1-sigmoid(z));

% =============================================================

end

% Testing your function 
g = sigmoidGradient([-1 -0.5 0 0.5 1])

function W = randInitializeWeights(L_in, L_out)

W = zeros(L_out, 1 + L_in);                                 % Return the following variables correctly 

% Randomly initialize the weights to small values
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end

% Initialize our weights
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

checkNNGradients;

%  Checking if your regularized NN is working 
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 50);
lambda = 1;                          %  You should also try different values of lambda
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

pred = predict(Theta1, Theta2, X);
Accuracy =  mean(double(pred == y)) * 100 

displayData(Theta1(:, 2:end));


