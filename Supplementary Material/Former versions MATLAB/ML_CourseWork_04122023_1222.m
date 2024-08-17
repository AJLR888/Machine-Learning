
% Load the data set
houses = readtable('houses.csv');

%%                      PARTITION into Train and Test                    %%

% For reproducibility, to reproduce always the same results.
rng('default')

% Partition the data set, reserve 30% for testing
c = cvpartition(height(houses), "holdout", 0.3);

% Split the data
train = houses(training(c), :);
test = houses(test(c), :);

%%                      LINEAR REGRESSION MODEL                          %%

% Train the model
lrm = fitlm(train, "PredictorVars",["totalBath", "BedroomAbvGr", "KitchenAbvGr"...
"Fireplaces", "GarageCars", "totalSpcBlt_standardized"...
"LotArea_standardized", "GrLivArea_normalized", "Neighborhood_BrkSide"...
"Neighborhood_ClearCr", "Neighborhood_CollgCr", "Neighborhood_Crawfor"...
"Neighborhood_Edwards", "Neighborhood_Gilbert", "Neighborhood_IDOTRR"...
"Neighborhood_MeadowV", "Neighborhood_Mitchel", "Neighborhood_NAmes"...
"Neighborhood_NPkVill", "Neighborhood_NWAmes", "Neighborhood_NoRidge"...
"Neighborhood_NridgHt", "Neighborhood_OldTown", "Neighborhood_SWISU"...
"Neighborhood_Sawyer", "Neighborhood_SawyerW", "Neighborhood_Somerst"...
"Neighborhood_StoneBr", "Neighborhood_Timber", "Neighborhood_Veenker"...
"BsmtFinType1_BLQ", "BsmtFinType1_GLQ", "BsmtFinType1_LwQ"...
"BsmtFinType1_Rec", "BsmtFinType1_Unf"], "ResponseVar", "SalePrice");

% Evaluate the training model, display metrics:
    % Anova
anova(lrm, 'summary')

    % Display MAE, RMSE, R-squared of the training model.
disp(lrm)

% Test the model
        % Define variables: 
xTest = test(:, ["totalBath", "BedroomAbvGr", "KitchenAbvGr"...
"Fireplaces", "GarageCars", "totalSpcBlt_standardized"...
"LotArea_standardized", "GrLivArea_normalized", "Neighborhood_BrkSide"...
"Neighborhood_ClearCr", "Neighborhood_CollgCr", "Neighborhood_Crawfor"...
"Neighborhood_Edwards", "Neighborhood_Gilbert", "Neighborhood_IDOTRR"...
"Neighborhood_MeadowV", "Neighborhood_Mitchel", "Neighborhood_NAmes"...
"Neighborhood_NPkVill", "Neighborhood_NWAmes", "Neighborhood_NoRidge"...
"Neighborhood_NridgHt", "Neighborhood_OldTown", "Neighborhood_SWISU"...
"Neighborhood_Sawyer", "Neighborhood_SawyerW", "Neighborhood_Somerst"...
"Neighborhood_StoneBr", "Neighborhood_Timber", "Neighborhood_Veenker"...
"BsmtFinType1_BLQ", "BsmtFinType1_GLQ", "BsmtFinType1_LwQ"...
"BsmtFinType1_Rec", "BsmtFinType1_Unf"]);

yTest = test.SalePrice;

% Make predictions for the testing model:
yPred = predict(lrm, xTest);

% Calculate relevant metrics for the testing model:
    % Residuals: 
residuals = yTest - yPred;

        % Plot residuals
figure;
scatter(yPred, residuals);
xlabel('Predicted Values');
ylabel('Residuals');
title('Residuals vs Predicted Values');

        % Adding horizontal line at zero:
hold on;
plot([min(yPred), max(yPred)], [0, 0], 'k--');
hold off;

  %%                      DECISION TREE MODEL

% Train the model
treeModel = fitctree(train,"SalePrice", 'MaxNumSplits', 4, 'MinLeafSize',10);

% Test the model
predValues = predict(treeModel, test);

% Display metrics
actualValues = test.SalePrice;
mse = mean((predValues - actualValues).^2);

disp(mse)
