
% Load the data set
houses = readtable('houses.csv');

%%                           1st BUILD THE MODELS                        %%

% For reproducibility, which means it store a random seed to get the same
% results in futures runnings.
rng('default')

% Partition 30/70
c = cvpartition(height(houses),"holdout",0.3);

% Split the data
train = houses(training(c), :);
test = houses(test(c), :);

% Dependent Variable
y = train.SalePrice;

% Independent variables, Excluding 'SalePrice' and 'Id' 
predictorNames = setdiff(train.Properties.VariableNames, {'SalePrice', 'Id'});
X = train{:, predictorNames};
%%
%  X is features matrix and y is the target variable
% Standardize the features before applying Lasso
[X, mu, sigma] = zscore(X);
%%
% Apply Lasso regression with 7-fold cross-validation
[B, FitInfo] = lasso(X, y, 'CV', 7);

% Looking the index with the best Lambda value
minMSEIndex = FitInfo.IndexMinMSE;
optimalLambda = FitInfo.Lambda(minMSEIndex);

% Coefficients for the optimal Lambda value
optimalCoefficients = B(:, minMSEIndex);

% Display the optimal Lambda and coefficients
disp(['Optimal Lambda: ', num2str(optimalLambda)]);
disp('Optimal Coefficients:');
disp(optimalCoefficients);
