houses = readtable('houses.csv');

% Shuffle the rows
houses_data = houses(randperm(size(houses,1)),:);


%% Create the Linear Regression model
lr_model = fitlm(x_train, y_train);

%% Making predictions on the test data
lr_predictions = predict(lr_model, x_test);

% Mean Squared Error (MSE) 
lr_MSE = immse(lr_predictions, y_test);

% R-squared value 
lr_Rsq = lr_model.Rsquared.Ordinary;

% Display the results
fprintf('Linear Regression MSE: %f\n', lr_MSE);
fprintf('Linear Regression R-squared: %f\n', lr_Rsq);



