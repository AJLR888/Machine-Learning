% Load the data set
houses = readtable('houses.csv');

%% 1st BUILD THE MODELS  %%

% For reproducibility, which means it store a random seed to get the same
% results in futures runnings.
rng('default')

% 70/30
c = cvpartition(height(houses),"holdout",0.3);

% Split the data
train = houses(training(c), :);
test = houses(test(c), :);

%% Building Linear Regression Model.%% 


% Show results
X_new = test{:, predictorNames}; 

% Predict using the trained model 'lrm'
y_pred = predict(lrm, X_new);


% Plotting residuals 
X = train{ :,["totalBath", "BedroomAbvGr"...
    "KitchenAbvGr", "Fireplaces", "GarageCars", "totalSpcBlt_standardized", "LotArea_standardized"...
    "GrLivArea_normalized", "Foundation_CBlock", "Foundation_PConc"... 
    "Foundation_Slab", "Foundation_Stone", "Foundation_Wood", "MasVnrType_Stone"...
    "Exterior2nd_AsphShn", "Exterior2nd_BrkFace", "Exterior2nd_ImStucc"... 
    "Exterior2nd_Other", "Exterior2nd_Stone", "Exterior2nd_Stucco"...
    "Exterior2nd_WdShng", "Exterior1st_AsphShn", "Exterior1st_ImStucc"...
    "Exterior1st_Stone", "Exterior1st_Stucco", "Exterior1st_WdShing"...
    "Neighborhood_BrkSide", "Neighborhood_ClearCr", "Neighborhood_Crawfor"... 
    "Neighborhood_Edwards", "Neighborhood_Gilbert", "Neighborhood_IDOTRR"...
    "Neighborhood_MeadowV", "Neighborhood_Mitchel", "Neighborhood_NPkVill"...
    "Neighborhood_NWAmes", "Neighborhood_NoRidge", "Neighborhood_NridgHt"...
    "Neighborhood_OldTown", "Neighborhood_SWISU", "Neighborhood_Sawyer"...
    "Neighborhood_SawyerW", "Neighborhood_Somerst", "Neighborhood_StoneBr"... 
    "Neighborhood_Timber", "Neighborhood_Veenker", "BsmtFinType1_BLQ"...
    "BsmtFinType1_GLQ", "BsmtFinType1_LwQ", "BsmtFinType1_Rec", "BsmtFinType1_Unf"...
    "GarageType_Basment", "GarageType_BuiltIn", "GarageType_CarPort"...
    "GarageFinish_RFn", "GarageFinish_Unf"]};

y_pred = predict(lrm, X);
residuals = lrm.Residuals.Raw;
scatter(y_pred, residuals)
xlabel('Predicted Values')
ylabel('Residuals')
title('Residuals vs. Predicted Values')


disp(length(lrm.PredictorNames));

disp(size(trainingData, 2));

disp(isprop(lrm, 'PredictorNames'));