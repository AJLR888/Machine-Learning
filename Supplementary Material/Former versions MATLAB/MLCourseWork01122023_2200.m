% Load the data set
houses = readtable('houses.csv');

%% 1st BUILD THE MODELS  %%

% For reproducibility, which means it store a random seed to get the same
% results in futures runnings.
rng('default')

% Partition the data set, reserve 30% for testing.
c = cvpartition(height(houses),"holdout",0.3);

% Split the data
train = houses(training(c), :);
test = houses(test(c), :);

%% Define variables:

% Response vector Y (dependent)
Y = train.SalePrice;

% Predictor matrix X (independent variables)
% Excluding 'SalePrice' and 'Id' 
predictorNames = setdiff(train.Properties.VariableNames, {'SalePrice', 'Id'});
X = train{:, predictorNames};

%%                           Linear Regression Model                     %% 

% Train the model
lrm = fitlm(X,Y);

% Display properties of the model
disp(lrm)

%% Ploting results
% Exclude 'Id' and 'SalePrice'  as well

lrm_predPrices = predict(lrm, X);
residuals = lrm.Residuals.Raw;
scatter(lrm_predPrices, residuals)
xlabel('Predicted Values')
ylabel('Residuals')
title('Residuals vs. Predicted Values')



