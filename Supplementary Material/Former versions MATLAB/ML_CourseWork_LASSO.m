
% Load the data set
houses = readtable('houses.csv');

%%                      PARTITION into Train and Test                    %%

% For reproducibility, to reproduce always the same results.
rng('default')

% Partition 30/70
c = cvpartition(height(houses), "holdout", 0.3);

% Split the data
train = houses(training(c), :);
test = houses(test(c), :);

%%             Lasso regression to select the best subsect of variables

% Creation of Varaibles:
    % Predictor matrix X (independent variables)
    % Excluding 'SalePrice' and 'Id' 
predictorNames = setdiff(train.Properties.VariableNames, {'SalePrice', 'Id'});
X = train{:, predictorNames};

% Response vector Y (dependent)
y = train.SalePrice;

% Standardize the features before applying Lasso
[X, mu, sigma] = zscore(X);

% Lasso regression for feature selection and avoid overfitting
    % with 7-fold cross-validation
[B, FitInfo] = lasso(X, y, 'CV', 7);

% Find the index of the Lambda value that gives the minimum cross-validated MSE
% In other words, find the best hypeparameter (Lambda)
minMSEIndex = FitInfo.IndexMinMSE;
optimalLambda = FitInfo.Lambda(minMSEIndex);

% Coefficients for the optimal Lambda value
optimalCoefficients = B(:, minMSEIndex);

% Display the optimal Lambda and non-zero coefficients with feature names
% This features will be used in the final Lineal Regression model
disp(['Optimal Lambda: ', num2str(optimalLambda)]);
disp('Optimal Coefficients with Feature Names:');

% Find indices of non-zero coefficients
nonZeroIndices = find(optimalCoefficients);

% Loop through non-zero indices to display feature names and coefficients
for idx = 1:numel(nonZeroIndices)
    featureIdx = nonZeroIndices(idx);
    fprintf('%s: %f\n', predictorNames{featureIdx}, optimalCoefficients(featureIdx));
end


