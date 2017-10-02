%% TUTORIAL - Titanic Survivor Predictions
% This MAIN file is used to help predict the survivors of the Titanic.
%
%% Initialization
clear ; close all; clc

%% Load Data
%  Load and parse
data_train_titanic = importfile1('train_titanic.txt', 2, 419);
data_test_titanic = importfile('test_titanic.txt', 2, 419);

X = data_train_titanic(:, [1, 3:end]); y = data_train_titanic(:, 2);
Xtest = data_test_titanic;

% Shuffle examples (good practice)
X = X(randperm(size(X,1)),:); y = y(randperm(size(y,1)),:);
Xtest = Xtest(randperm(size(Xtest,1)),:);

% Remove features
X = X(:, [1:2, 4:9, 11]); 
Xtest = Xtest(:, [1:2, 4:9, 11]);

% Compute Xtrain, ytrain from percentage of cv desired
percentage = 20;
rowper = floor(size(X,1) * percentage/100);
Xval = X(1:rowper,:); yval = y(1:rowper,1);
Xtrain = X(rowper+1:end,:); ytrain = y(rowper+1:end,1);

% Normalize features
[X_norm, mu, sigma] = featureNormalize(Xtrain);



%% NEED TO DO - 
% 1. Use minimal features without reg to see if you can fit the data
% (remove Age NaN's with mean)
% 2. If high bias, add more features. Might need to map with poly's, too.
% 3. Ensure model isn't too varied by optimizing lambda with learning
% curve.
% 4. Calculate test error to see if it's near cost of train/cv