
warning('off'); addpath('../readonly/Assignment3/');
load('ex3data1.mat');     % X and y are arrays where your training data is stores
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
m = size(X, 1);           % number of rows

whos;                     % Check out your new variables

rand_indices = randperm(m);          % Randomly select 100 data points to display
sel = X(rand_indices(1:100), :);     
displayData(sel);                    

% GRADED FUNCTION: lrCostFunction
function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);                        % number of training examples
J = 0;                                % return the following variables correctly 
grad = zeros(size(theta));            % initializing the gradient

% ====================== YOUR CODE HERE ======================
h = sigmoid(X * theta);
J = -1/m * (y'*log(h) + (1-y)'*log(1-h)) + lambda/(2*m) * (theta(2:end)'*theta(2:end));
grad(1,1) = 1/m * (X(:,1)'*(h-y));
grad(2:end) = 1/m *(X(:,2:end)'*(h-y)) + (lambda/m) *theta(2:end);
% ============================================================

grad = grad(:);

end

% Initialize random variables to check your implementation
lambda = 0.9;                           % Set this to 0.9 to see if your output matches ours for the regularized part.
temp_X = X(30:34,130:134);                                   % Get a random X matrix
temp_theta = [3,-5,13,-.4, 0.3]';                            % Initialize theta
temp_y = [1,2,-3,4,-5]';                                     % Initialize y  
                                                            
% Compute Cost and gradient
[J, grad] = lrCostFunction(temp_theta, temp_X, temp_y, lambda);

% Check your results! 
% We only printed the first few grad numbers to avoid printing the entire list
J 
grad(1:5)

a = 1:10 % Create a and b 
b = 3
a == b    % You should try different values of b here

% GRADED FUNCTION: oneVsAll
function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);                                      % Some useful variables
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);                % You need to return the following variable correctly
X = [ones(m, 1) X];                                  % Add ones to the X data matrix

% ====================== YOUR CODE HERE ======================
for i = 1:num_labels
    initial_theta = zeros(n+1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)),initial_theta, options);
    all_theta(i,:) = theta(:);
end
% =============================================================
end

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

% GRADED FUNCTION: predictOneVsAll
function p = predictOneVsAll(all_theta, X)

m = size(X, 1);                                 % Declaring some useful variables
num_labels = size(all_theta, 1);

p = zeros(m, 1);                       % Return the following variables correctly 
X = [ones(m, 1) X];                             % Add ones to the X data matrix

% ====================== YOUR CODE HERE ======================
h = sigmoid(X * all_theta');
[x, i] = max(h,[], 2);
p = i;
% ============================================================
end

whos
pred = predictOneVsAll(all_theta, X);
Accuracy =  mean(double(pred == y)) * 100

load('ex3data1.mat');           % loading the data
load('ex3weights.mat');
m = size(X, 1);                 % number of rows

sel = randperm(size(X, 1));     % randomly select 100 data points to display
sel = sel(1:100);

displayData(X(sel, :));         % display the data 

% GRADED FUNCTION: predict
function p = predict(Theta1, Theta2, X)

% X is 5000 by 400
% Theta1 is 25 by 401
% Theta2 is 10 by 26

m = size(X, 1);                                % Useful values
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);                      % Return the following variable correctly 

% ====================== YOUR CODE HERE ======================
X = [ones(m,1) X];
a2 = sigmoid(X * Theta1');
a2 = [ones(size(a2,1),1) a2];

h = sigmoid(a2 * Theta2');
[x, i] = max(h, [], 2);
p = i;

% ============================================================

end

% Now we will call your predict function using the 
% loaded set of parameters for Theta1 and Theta2. 
pred = predict(Theta1, Theta2, X);         % predicting the output
Accuracy = mean(double(pred == y)) * 100   % Accuracy 

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Keep re-running this cell to see different numbers 
k = 10000;  
%  Randomly permute examples one at a time
rp = randperm(k/2);

% Display 
fprintf('\nDisplaying Example Image\n');
displayData(X(rp(k/2), :));

pred = predict(Theta1, Theta2, X(rp(k/2),:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, k/2));
