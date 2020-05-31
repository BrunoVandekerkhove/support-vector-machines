%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
addpath(genpath('svm'))
close all
%% Load dataset
load('diabetes')
Xtrain = total;
Ytrain = labels_total;
Xtest = testset;
Ytest = labels_test;
Ytrain(Ytrain == 1) = 'P';
Ytest(Ytest == 1) = 'P';
Ytrain(Ytrain == -1) = 'N';
Ytest(Ytest == -1) = 'N';
Ytrain = categorical(Ytrain);
Ytest = categorical(Ytest);
%% NN approach
c = cvpartition(Ytrain, 'HoldOut', 0.5, 'Stratify', true);
XtrainMLP = Xtrain(c.training,:)';
YtrainMLP = Ytrain(c.training)';
XvalMLP = Xtrain(c.test,:)';
YvalMLP = Ytrain(c.test)';
layers = [
    sequenceInputLayer(8, 'Normalization', 'zerocenter')
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(8)
    reluLayer
    fullyConnectedLayer(8)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    %sigmoidLayer
    %maeRegressionLayer()
];
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 1000, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XvalMLP,YvalMLP}, ...
    'ValidationFrequency', 1, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
net = trainNetwork(XtrainMLP, YtrainMLP, layers, options);
%% MLP prediction
t = 0.5;
Output = predict(net, Xtest');
Ypred = zeros(length(labels_test),1);
Ypred(Output(1,:) > t) = -1;
Ypred(Output(1,:) <= t) = 1;
fprintf('Accuracy = %.5f\n', sum(Ypred == labels_test)/length(Ytest));

