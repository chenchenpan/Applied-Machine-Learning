
% Load the Data
warning('off'); addpath('../readonly/Assignment1b/');
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% GRADED FUNCTION: featureNormalize
function [X_norm, mu, sigma] = featureNormalize(X)

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
mu = mean(X);
sigma = std(X);
# X_norm = bsxfun(@rdivide, (X - mu), sigma);
X_norm = (X - mu) ./ std(X);
% ============================================================
end

% Scale features and set them to zero mean
[X mu sigma] = featureNormalize(X)

% Add intercept term to X
X = [ones(m, 1) X];

% GRADED FUNCTION: computeCostMulti
function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples
J = 0;         % compute the cost and set it to J

% ====================== YOUR CODE HERE ======================
J = 1/(2*m) * (X * theta - y)' * (X * theta - y);

% ============================================================
end

% Load data
data = load('ex1data2.txt'); X = data(:, 1:2); y = data(:, 3); m = length(y);

% Normalize and add ones
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];

% Compute cost for theta with all zeros
theta = zeros(3, 1);         % We already loaded X and y

computeCostMulti(X, y, theta)

% Load variables
alpha = 0.01;                % Initializing alpha
num_iters = 400;             % Number of iterations 
whos                         % The list of variables should include X, y, theta, alpha and num_iters

% GRADED FUNCTION: gradientDescentMulti
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y);                    % number of training examples
J_history = zeros(num_iters, 1);  % vector to store the cost at every iteration

    for iter = 1:num_iters

    % ====================== YOUR CODE HERE =====================

    theta -= alpha/m * X' * (X*theta - y);



    % ===========================================================
    J_history(iter) = computeCostMulti(X, y, theta); % Save the cost J in every iteration    

    end

end

% Initialize some variables
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters); % run your gradient descent

% Plot the convergence graph
figure('Position',[0,0,400,400]);
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

theta      % Display result from gradient descent

% ====================== YOUR CODE HERE ======================
x = bsxfun(@rdivide, ([1650 3] - mu), sigma);
price = [1 x] * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

% Do it for different alpha values. (i.e. initiialize different alphas)
alpha_1 = 0.02;
num_iters = 400;


% Add some code below to run gradientDescentMulit on different alphas and thetas
% You could initialize J_2, J_3, etc.. the same way we have have J_1 
theta_1 = zeros(3, 1);
theta_2 = zeros(3, 1);
theta_3 = zeros(3, 1);
[theta_1, J_1] = gradientDescentMulti(X, y, theta_1, alpha_1, num_iters);
[theta_2, J_2] = gradientDescentMulti(X, y, theta_2, 0.0006, num_iters);
[theta_3, J_3] = gradientDescentMulti(X, y, theta_3, 0.09, num_iters);
% ----------------------------------------------------------
% Plot the convergence graphs
figure('Position',[0,0,400,400]);
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% To compare how different learning learning rates affect convergence, 
% it's helpful to plot J for several learning rates on the same figure.

hold on;
plot(1:numel(J_1), J_1, 'r', 'LineWidth',2);
hold on;
plot(1:numel(J_2), J_2(1:400), 'g');
plot(1:numel(J_3), J_3(1:400), 'k');

% The final arguments 'b', 'r', and 'k' specify different colors for the plots.


% GRADED FUNCTION: normalEqn
function [theta] = normalEqn(X, y)

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================

theta = pinv(X'*X) * X' * y;

% ============================================================

end

# data = load('ex1data2.txt'); X = data(:, 1:2); y = data(:, 3);
theta = normalEqn(X, y)

% Try it below
x2 = bsxfun(@rdivide, ([1650 3] - mu), sigma);
price = [1 x2] * theta

cost = computeCostMulti(X, y, theta)

theta1 = [3.3430e+05 9.9411e+04 3.2670e+03]

cost = computeCostMulti(X, y, theta1')


