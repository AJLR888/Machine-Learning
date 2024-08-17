% load the data set

houses_data = readtable('houses.csv');

% Create a table with the target variable and the features
houses_data =  Id;SalePrice,totalSpcBlt_standardized, totalBath, 
Neighborhood_Veenker, Neighborhood_Timber, Neighborhood_SWISU, 
Neighborhood_StoneBr, Neighborhood_Somerst, Neighborhood_SawyerW, 
Neighborhood_Sawyer, Neighborhood_OldTown, Neighborhood_NWAmes, 
Neighborhood_NridgHt, Neighborhood_NPkVill, Neighborhood_NoRidge, 
Neighborhood_NAmes, Neighborhood_Mitchel, Neighborhood_MeadowV, 
Neighborhood_IDOTRR, Neighborhood_Gilbert, Neighborhood_Edwards, 
Neighborhood_Crawfor, Neighborhood_CollgCr, Neighborhood_ClearCr,
Neighborhood_BrkSide, Neighborhood_Blmngtn, MasVnrType_Stone, 
MasVnrType_None, MasVnrType_BrkFace, MasVnrType_BrkCmn,
LotArea_standardized, KitchenAbvGr, GrLivArea_normalized,
GrLivArea, GarageType_Detchd, GarageType_CarPort, GarageType_BuiltIn,
GarageType_Basment, GarageType_Attchd, GarageType_2Types, GarageFinish_Unf,
GarageFinish_RFn, GarageFinish_Fin, GarageCars, Foundation_Wood,
Foundation_Stone, Foundation_Slab, Foundation_PConc, Foundation_CBlock,
Foundation_BrkTil, Fireplaces, Exterior2nd_VinylSd, Exterior2nd_Stucco, 
Exterior2nd_Stone, Exterior2nd_Plywood, Exterior2nd_Other,
Exterior2nd_MetalSd, Exterior2nd_ImStucc, Exterior2nd_HdBoard,
Exterior2nd_CmentBd, Exterior2nd_CBlock, Exterior2nd_BrkFace,
Exterior2nd_AsphShn, Exterior2nd_AsbShng, Exterior1st_WdShing, 
Exterior1st_VinylSd, Exterior1st_Stucco, Exterior1st_Stone,
Exterior1st_Plywood, Exterior1st_MetalSd, Exterior1st_ImStucc, 
Exterior1st_HdBoard, Exterior1st_CemntBd, Exterior1st_CBlock, 
Exterior1st_BrkFace, Exterior1st_AsphShn, Exterior1st_AsbShng, 
BsmtFinType1_Unf, BsmtFinType1_Rec, BsmtFinType1_LwQ, BsmtFinType1_GLQ,
BsmtFinType1_BLQ, BsmtFinType1_ALQ, BedroomAbvGr;

% Shuffle the rows
houses_data = houses_data(randperm(size(houses_data,1)),:);

% 80-20 division
splitPoint = floor(0.8 * height(houses_data));

% Split the data into the two data sets
train = houses_data(1:splitPoint, :);
test = houses_data(splitPoint+1:end, :);

% Defining predictors/dependent v.
x_train = train(:, setdiff(train.Properties.VariableNames, {'SalePrice'}));
y_train = train.SalePrice;

% Create the Linear Regreassion model
lr_model = fitlm(train(:, setdiff(train.Properties.VariableNames, {'SalePrice'})), train.SalePrice);

% Test data
x_test = test(:, setdiff(test.Properties.VariableNames, {'salePrice'}));
y_test = test.SalePrice;

% Running and checking metrics
lr_predictions = predict(lr_model, x_test);
lr_MSE = immse(lr_predictions, y_test);
lr_Rsq = lr_model.Rsquared.Ordinary;
 

% DECISION TREE MODEL:

% Create the model
tree_model = fitrtree(x_train, y_train);

% Running and checking metrics
tree_predictions = predict(tree_model, x_test);
tree_MSE = immse(tree_predictions, y_test);
tree_Rsq = tree_model.Rsquared.Ordinary;


