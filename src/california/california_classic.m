%% Initialisation
clc
clear
addpath(genpath('./figures'))
addpath(genpath('./lssvm'))
addpath(genpath('./data'))
addpath(genpath('./california'))
close all
%% Load data
data = load('california.dat', '-ascii');
data = data(1:160,:);
normal_price_indices = data(:,end) < 500000;
high_price_indices = data(:,end) >= 500000;
%data = data(normal_price_indices, :);
%data = data(1:10000,:); % take subset
X = data(:,1:end-1);
Y = data(:,end);
trainsize = ceil(3*length(Y)/4); % stratified sampling would be better
Xtrain = X(1:trainsize,:);
Ytrain = Y(1:trainsize,:);
Xtest = X(trainsize+1:end,:);
Ytest = Y(trainsize+1:end,:);
[Xtrain,Ytrain,Xtest,Ytest] = initial(Xtrain, Ytrain, 'f', Xtest, Ytest);
%%
[nsv,beta,bias] = svr(Xtrain, Ytrain, 'anovaspline3', 1000, 'eInsensitive', 1.85);
%%
predictedY = svroutput(Xtrain,Xtest,'anovaspline3',beta,bias);