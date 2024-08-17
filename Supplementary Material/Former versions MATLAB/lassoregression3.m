
%%        Lasso regression    %%
% Load the data set
houses = readtable('houses.csv');

% For reproducibility
rng('default')

% Partition 70/30
c = cvpartition(height(houses), "holdout", 0.3);

% Split the data
train = houses(training(c), :);
test = houses(test(c), :);

% Dependent variable
y = train.SalePrice;

% Independent variables
% Excluding 'SalePrice' and 'Id' 
predictorNames = setdiff(train.Properties.VariableNames, {'SalePrice', 'Id'});
X = train{:, predictorNames};
%%
% Standardize the features before applying Lasso
[X, mu, sigma] = zscore(X);

% Apply Lasso regression with 7-fold cross-validation
[B, FitInfo] = lasso(X, y, 'CV', 7);

% Find the index of the Lambda value that gives the minimum cross-validated MSE
minMSEIndex = FitInfo.IndexMinMSE;
optimalLambda = FitInfo.Lambda(minMSEIndex);

% Coefficients for the optimal Lambda value
optimalCoefficients = B(:, minMSEIndex);

%%
% Optimal Lambda and non-zero coefficients with feature names
disp(['Optimal Lambda: ', num2str(optimalLambda)]);
disp('Optimal Coefficients with Feature Names:');

% Find indices of non-zero coefficients
nonZeroIndices = find(optimalCoefficients);

% Loop through non-zero indices to display feature names and coefficients
for idx = 1:numel(nonZeroIndices)
    featureIdx = nonZeroIndices(idx);
    fprintf('%s: %f\n', predictorNames{featureIdx}, optimalCoefficients(featureIdx));
end

%%                      LINEAR REGRESSION SELECTED FEATURES             %%

