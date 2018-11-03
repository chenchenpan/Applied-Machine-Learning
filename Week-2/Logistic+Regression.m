
% Load the Data 
warning('off'); addpath('../readonly/Assignment2');
data = load('ex2data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);
whos

function plotData(X, y)
% Create New Figure
figure; hold on;
pos = find(y==1);           % Find Indices of Positive Examples
neg = find(y == 0);         % Find Indices of Negative Examples

%Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7); 
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);
hold off;
end

plotData(X,y)              
hold on;                        % We purposely didn't put the xlabel, ylabel commands in the function
xlabel("Exam Score 1")          % So that you could use your function again in the second part of the exercise below
ylabel("Exam Score 2")

% GRADED FUNCTION: sigmoid
function g = sigmoid(z)

g = zeros(size(z));     % return this correctly

% ====================== YOUR CODE HERE ======================
g = rdivide(1, 1+exp(-z));


% =============================================================

end

sigmoid(0)
sigmoid(100)

% GRADED FUNCTION: costFunction
function [J, grad] = costFunction(theta, X, y)

m = length(y);                   % number of training examples


J = 0;                           % set it correctly
grad = zeros(size(theta));       % set it correctly

% ====================== YOUR CODE HERE ======================
h = sigmoid(X * theta);
J = -1/m * ((y' * log(h)) + ((1-y)' * log(1-h)));
grad = 1/m * (X' * (h-y));
% ============================================================

end

[m, n] = size(X);                      % Setup the data matrix appropriately
X = [ones(m, 1) X];                    % Add intercept term 
initial_theta = zeros(n + 1, 1);       % Initialize fitting parameters
[cost, grad] = costFunction(initial_theta,X,y)
cost

% Run Optimset                     
options = optimset('GradObj', 'on', 'MaxIter', 400);    %  Set options for fminunc

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options)

% Plot the Decision Boundary 
plotDecisionBoundary(theta, X, y);       

prob = sigmoid([1 45 85] * theta)

% GRADED FUNCTION: predict
function p = predict(theta, X)

m = size(X, 1);                     % Number of training examples
p = zeros(m, 1);                    % Return the following variable correctly

% ====================== YOUR CODE HERE ======================
p = sigmoid(X * theta) >= 0.5;

% =============================================================

end

p = predict(theta, X);                       % Calling your function
Accuracy = mean(double(p == y)) * 100        % Calculating the Accuracy

% Load the Variables
data = load('ex2data2.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

plotData(X,y)
hold on;                                              
xlabel('Microchip Test 1')      % Labels and Legend
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;

function out = mapFeature(X1, X2)

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end

% GRADED FUNCTION: costFunctionReg
function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);                % Number of training examples
J = 0;                        % Set J to the cost
grad = zeros(size(theta));    % Set grad to the gradient

% ====================== YOUR CODE HERE ======================
h = sigmoid(X*theta);
J = 1/m * (-y'*log(h) - (1-y)'*log(1-h)) + lambda/(2*m)*(theta(2:end)'* theta(2:end));
grad(1,1) = 1/m*(X(:,1)' * (h-y));
grad(2:end) = 1/m*(X(:,2:end)' * (h-y)) + lambda/m *theta(2:end);

% =============================================================

end

X = mapFeature(X(:,1), X(:,2));                            % Add Polynomial Features (it adds the intercept term)
initial_theta = zeros(size(X, 2), 1);                      % Initialize fitting parameters
lambda = 1;                                                % Set regularization parameter lambda to 1
cost = costFunctionReg(initial_theta, X, y, lambda)

initial_theta = zeros(size(X, 2), 1);                % Initialize fitting parameters
lambda = 1;                                          % Set regularization parameter lambda to 1 (you should vary this)
options = optimset('GradObj', 'on', 'MaxIter', 400); % Set Options


[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
J

% Plot the Decision Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))             % Labels and Legend    
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

p = predict(theta, X);                            % Get the prediction
Accuracy =  mean(double(p == y)) * 100            % Accuracy 


