%% Initialisation
clc
clear
addpath(genpath('./figures'))
addpath(genpath('./lssvm'))
addpath(genpath('./data'))
close all
%% Read data
data = load('shuttle.dat', '-ascii'); 
data = data(1:58000,:);
%% Set up data
classes_to_keep = 1:5;
indices_to_keep = ismember(data(:,end), classes_to_keep);
use_testset = 1;
X = data(indices_to_keep,1:end-1);
Y = data(indices_to_keep,end);
if use_testset % set to 1 to do multiclass classification with test set
    % Build train / val / test (random split, stratified)
    c = cvpartition(Y, 'HoldOut', 0.25, 'Stratify', true);
    Xtrain = X(c.training,:);
    Ytrain = Y(c.training);
    Xtest = X(c.test,:);
    Ytest = Y(c.test);
    fprintf('Class imbalance : %.5f\n', sum(Ytest==1) / length(Ytest));
    for cl = 1:7
        fprintf('# (class = %d) : %d\n', cl, sum(Ytest==cl));
    end
else
    Xtrain = X;
    Ytrain = Y;
    Xtest = [];
    Ytest = [];
end
%% Do KNN
neighbours = 2;
nns = knnsearch(Xtrain, Xtest, 'K', neighbours);
%% Evaluate
errors = sum(Ytest ~= Ytrain(mode(nns,2)));
accuracy = 1 - errors/length(Ytest) %#ok