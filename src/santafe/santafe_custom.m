%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
addpath(genpath('svm'))
close all
%% Load dataset and prepare it
load('santafe')
order = 10;
X = windowize(Z, 1:(order + 1));
Y = X(:, end);
X = X(:, 1:order);
%% Training
gamma = 1.0;
sigma = 1.0;
degree = 2;
[alpha,b] = trainlssvm({X, Y, 'f', gamma, [sigma,degree], 'ANOVA_kernel'});
%% Prediction
to_predict = Ztest;
Xs = Z(end-order+1:end,1);
nb = length(to_predict);
prediction = predict({X, Y, 'f', gamma, sigma, 'RBF_kernel'}, Xs, nb);
% Visualise
figure;
hold on;
n=plot(to_predict, 'k', 'LineWidth', 2);
n.set('Color', [0, 0.4470, 0.7410]);
p=plot(prediction, 'r', 'LineWidth', 2);
p.set('Color', [0.6350, 0.0780, 0.1840]);
hold off;