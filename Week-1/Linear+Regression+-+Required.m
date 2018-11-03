
% GRADED FUNCTION: warmUpExercise 
function A = warmUpExercise()
A = [];
% ============= YOUR CODE HERE ==============
A = eye(5)
% ===========================================
end

A = warmUpExercise()

warning('off'); addpath('../readonly/Assignment1a/');  % Add a path to the files                
data = load('ex1data1.txt');
X = data(:, 1);                       % population of a city
y = data(:, 2);                       % profit of a food truck in the city
m = length(y);                        % number of training examples

% Try out the 'whos' function and the 'help plot' functions.
% Everytime you need to look at some documentation or see your variables you could run this cell. 
%-----------------------------------------------
whos

function plotData(x, y)
% Expected to return a figure of this size.
figure('Position', [0,0,400,400]);
plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s')
xlabel('Population of City in 10,000s')
end

% Plot the Data
data = load('ex1data1.txt'); X = data(:, 1); y = data(:, 2);
plotData(X, y)

% GRADED FUNCTION: computeCost 
function J = computeCost(X, y, theta)

m = length(y); % number of training examples - a useful value
J = 0;         % return the following varaible correctly. 

% ====================== YOUR CODE HERE ======================
J = 1/(2*m) * sum((X*theta - y).^2);

% ============================================================
end

% Load data and initialize variables
data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y);

X = [ones(m, 1), data(:,1)];             % Add a column of ones to X to account for the intercept
theta = zeros(2, 1);                     % initialize fitting parameters

% Gradient descent settings
iterations = 1500;
alpha = 0.01;

computeCost(X, y, theta) % Compute and display the initial cost

% GRADED FUNCTION: gradientDescent 
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y);                   % number of training examples
J_history = zeros(num_iters, 1); % a vector to save the cost of J in every iteration

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    theta -= alpha/m * (X' * (X*theta - y));

    
    % ============================================================ 
    J_history(iter) = computeCost(X, y, theta);     % Save the cost J in every iteration 
end

end

theta = gradientDescent(X, y, theta, alpha, iterations)

plotData(X(:,2),y)
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression','Location','southeast')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Surface plot
figure('Position',[0,0,1000,400]);
subplot (1, 2, 1)
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0',"fontsize", 20); ylabel('\theta_1',"fontsize", 20);

% Contour plot
subplot (1, 2, 2)
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0',"fontsize", 20); ylabel('\theta_1',"fontsize", 20);
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


