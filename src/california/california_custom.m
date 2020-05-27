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
%% Visualise high prices
normal_price_indices = data(:,end) < 500000;
high_price_indices = data(:,end) >= 500000;
% hold on
% p = plot(data(normal_price_indices,1), data(normal_price_indices,2), 'b.');
% set(p, 'Color', [0, 0.4470, 0.7410]);
% p = plot(data(high_price_indices,1), data(high_price_indices,2), 'r.');
% set(p, 'Color', [0.6740, 0.2, 0.1880]);
%% Prepare data
%data = data(normal_price_indices, :);
%data = data(1:10000,:); % take subset
X = data(:,1:end-1);
Y = data(:,end);
trainsize = ceil(3*length(Y)/4); % stratified sampling would be better
Xtrain = X(1:trainsize,:);
Ytrain = Y(1:trainsize,:);
Xtest = X(trainsize+1:end,:);
Ytest = Y(trainsize+1:end,:);
[Xtrain,~,Xtest,~] = initial(Xtrain, Ytrain, 'f', Xtest, Ytest);
%% Correlation Coefficients
for f = 1:9
    matrix = corrcoef(data(:,f),data(:,end));
    fprintf('Correlation %d = %.5f\n', f, matrix(1,2));
end
%% Histograms (all samples)
for f = 1:8
    figure
    histogram(X(:,f))
end
%% Histograms (train/test set)
for i = 1:8
    figure
    hold on
    histogram(Xtrain(:,i))
    histogram(Xtest(:,i))
    hold off
    drawnow
end
figure
hold on
histogram(Ytrain)
histogram(Ytest)
hold off
drawnow
%% ARD (doesn't filter anything)
% tune_params = {Xtrain, Ytrain, 'f', 0.1, [], 'RBF_kernel'};
% [gam,sig2,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'mse'});
% fprintf('Tuning results (g,s,c) = (%.3f,%.3f,%.3f)\n', gam, sig2, cost)
% [selected, ranking] = bay_lssvmARD({Xtrain, Ytrain, 'c', gam, sig2}) %#ok
% save('california/california_ard', 'selected', 'ranking') % save intermediate results
% load('california/california_ard') % load intermediate results
% Xtrain = Xtrain(:,:); % relevant inputs of training set
% Xtest = Xtest(:,:); % relevant inputs of test set
%% FS LS-SVM
global repetitions
repetitions = 1;
k = 2;
kernel_type = 'poly_kernel'; % or 'lin_kernel', 'poly_kernel'
user_process={'FS-LSSVM'}; %, 'SV_L0_norm'
window = [15,20,25];
% Tune/Train/Test
[e,s,t] = fslssvm(Xtrain, Ytrain, k, 'f', kernel_type, 'csa', user_process, window, Xtest, Ytest);