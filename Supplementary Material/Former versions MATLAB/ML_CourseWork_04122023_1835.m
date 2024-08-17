%%                                   Note:
% Dependent varaible = ResponseVar = SalePrice
% Note2:     lrmPred = Linear Regression Model, Prediction of Y (SalePrice)
%            lrmTest = Linear Regression Model, Test Data.
%           lrmTrain = Linear Regression Model, Train Data

%%                           Load the data set
houses = readtable('houses.csv');

%%                      PARTITION into Train and Test                    %%


% For reproducibility, to reproduce always the same results.
rng('default')

% Partition the data set, reserve 30% for testing
c = cvpartition(height(houses), "holdout", 0.3);

% Split the data
trainData = houses(training(c), :);
testData = houses(test(c), :);

% Lets ignore the "Id" column:
Train = removevars(trainData, 'Id');
Test = removevars(testData, 'Id');

%%                      LINEAR REGRESSION MODEL  (lrm)                        %%

% Train the model
lrm = fitlm(Train, "PredictorVars",["totalBath", "BedroomAbvGr", "KitchenAbvGr"...
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

%% Evaluate the training model, display metrics:
yTrain = Train.SalePrice;

lrmPredTrain = predict(lrm, Train);

lrmTrainMAE = mean(abs(yTrain - lrmPredTrain));

lrmTrainRMSE = sqrt(mean((yTrain - lrmPredTrain).^2));

lrmTrainRSquared = lrm.Rsquared.Ordinary;

        % Plot residuals
residuals = yTrain - lrmPredTrain;

figure;
scatter(lrmPredTrain, residuals);
xlabel('Predicted Train Values');
ylabel('Residuals');
title('Residuals vs Predicted Train Values');

            % Adding horizontal line at zero:
hold on;
plot([min(lrmPredTrain), max(lrmPredTrain)], [0, 0], 'k--');
hold off;
%% Evaluate the Linear Regression test model, display metrics:

yTest = Test.SalePrice;

lrmPredTest = predict(lrm, Test);

lrmTestMAE = mean(abs(yTest - lrmPredTest));

lrmTestRMSE = sqrt(mean((yTest - lrmPredTest).^2));

lrmTest_SSres = sum((yTest - lrmPredTest).^2);
lrmTest_SStot = sum((yTest - mean(yTest)).^2);
lrmTestRSquared = 1 - (ss_res / ss_tot);

        % Plot residuals
residuals = yTest - lrmPredTest;

figure;
scatter(lrmPredTest, residuals);
xlabel('Predicted Test Values');
ylabel('Residuals');
title('Residuals vs Predicted Test Values');

            % Adding horizontal line at zero:
hold on;
plot([min(lrmPredTest), max(lrmPredTest)], [0, 0], 'k--');
hold off;
%% Linear Regression Print Results:
fprintf("===============================================\n");
fprintf("Linear Regression Model Train Vs Test Results:\n");
fprintf("-----------------------------------------------\n");
fprintf('Mean Absolute Error (MAE): %.3f\n', lrmTrainMAE);
fprintf('Mean Absolute Error (MAE): %.3f\n', lrmTestMAE);
fprintf("-----------------------------------------------\n");
fprintf('Root Mean Squared Error (RMSE): %.3f\n', lrmTrainRMSE);
fprintf('Root Mean Squared Error (RMSE): %.3f\n', lrmTestRMSE);
fprintf("-----------------------------------------------\n");
fprintf('R-squared: %.3f\n', lrmTrainRSquared);
fprintf('R-squared: %.3f\n', lrmTestRSquared);

  %%                      DECISION TREE MODEL (dtm)

% Train the model
dtm = fitctree(train,"SalePrice", 'MaxNumSplits', 4, 'MinLeafSize',10);

% Evaluate the training model, display metrics:

dtmPredTrain = predict(dtm, Train);

dtmTrainMAE = mean(abs(yTrain - dtmPredTrain));

dtmTrainRMSE = sqrt(mean((yTrain - dtmPredTrain).^2));

dtmTrain_SSres = sum((yTrain - dtmPredTrain).^2);
dtmTrain_SStot = sum((yTrain - mean(yTrain)).^2);
dtmTrainRSquared = 1 - (dtmTrain_ss_res / dtmTrain_ss_tot);

%% Evaluate the Decision Tree test model, display metrics

dtmPredTest = predict(dtm, Test);

dtmTestMAE = mean(abs(yTest - dtmPredTest));

dtmTestRMSE = sqrt(mean((yTest - dtmPredTest).^2));

dtmTest_SSres = sum((yTest - lrmPredTest).^2);
dtmTest_SStot = sum((yTest - mean(yTest)).^2);
dtmTestRSquared = 1 - (dtmTest_ss_res / dtmTest_ss_tot);
 
%% Decision Tree Print Results:
fprintf("===============================================\n");
fprintf("Decision Tree Model Train Vs Test Results:\n");
fprintf("-----------------------------------------------\n");
fprintf('dtm Mean Absolute Error (MAE): %.3f\n', dtmTrainMAE);
fprintf('dtm Mean Absolute Error (MAE): %.3f\n', dtmTestMAE);
fprintf("-----------------------------------------------\n");
fprintf('dtm Root Mean Squared Error (RMSE): %.3f\n', dtmTrainRMSE);
fprintf('dtm Root Mean Squared Error (RMSE): %.3f\n', dtmTestRMSE);
fprintf("-----------------------------------------------\n");
fprintf('dtm R-squared: %.3f\n', dtmTrainRSquared);
fprintf('dtm R-squared: %.3f\n', dtmTestRSquared);

%%


















% Define variables:
    dtmPred = predict(dtm, )

% Evaluate the Training Model, display metrics: 
    % Mean Absolute Error (MAE)
    dtmMae = mean(abs(dtm - ))
    % Root Mean Squared Error (RMSE)

    % R- Squared


% Test the model by making prediction 


% Evaluate the Test Model, display metrics: 
    % Mean Absolute Error (MAE)
    

    % Root Mean Squared Error (RMSE)

    % R- Squared

% Display metrics 
    %
    % Make predictions on the training data
dtmPred = predict(dtree, X_train);

    % Mean Absolute Error (MAE)
mae = mean(abs(yTrain - lrmPred));
fprintf('Mean Absolute Error (MAE): %.3f\n', mae);
fprintf('Root Mean Squared Error (RMSE): %.3f\n', rmse);
fprintf('R-squared: %.3f\n', r_squared);

% Calculate Root Mean Squared Error (RMSE)
rmse = sqrt(mean((y_train - lrmPred).^2));
fprintf('Root Mean Squared Error (RMSE): %.3f\n', rmse);

% Calculate R-squared
SStotal = sum((y_train - mean(y_train)).^2);
SSres = sum((y_train - lrmPred).^2);
r_squared = 1 - SSres/SStotal;
fprintf('R-squared: %.3f\n', r_squared);

actualValues = test.SalePrice;
mse = mean((predValues - actualValues).^2);

disp(mse)


% Test the model by making predictions of the SalePrice
dtmPred = predict(dtm, xTest);
