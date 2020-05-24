%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
addpath(genpath('svm'))
close all
load('ripley')
X = Xtrain;
Y = Ytrain;
type = 'classification';
L_fold = 10; % L-fold crossvalidation
[gam,sig2] = tunelssvm({X,Y,type,1,1,'RBF_kernel'}, 'gridsearch','crossvalidatelssvm',{L_fold,'misclass'});
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
Y_latent = latentlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b},X);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,Y);
[thresholds oneMinusSpec Sens] %#ok