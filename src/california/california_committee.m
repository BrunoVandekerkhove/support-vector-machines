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
%data = data(1:1600,:); % select subset
sinc_demo = 0;
if sinc_demo
    rng('default')
    X = (-3:0.002:3)'; 
    X = X(randperm(length(X)));
    Y = sinc(X) + 0.1*randn(length(X), 1);
    Xtrain = X(1:1200,:);
    Ytrain = Y(1:1200);
    Xtest = X(1201:end,:);
    Ytest = Y(1201:end);
else
    X = data(:,1:end-1);
    Y = data(:,end);
    c = cvpartition(Y, 'HoldOut', 0.25, 'Stratify', true);
    Xtrain = X(c.training,:);
    Ytrain = Y(c.training);
    Xtest = X(c.test,:);
    Ytest = Y(c.test);
end
%% Visualise high prices
% Can be used to filter samples
normal_price_indices = data(:,end) < 500000;
high_price_indices = data(:,end) >= 500000;
%% Prepare data
preprocess = 1; % 1 = standardize, 2 = normalize
if preprocess == 1
    % zero mean and unit variance
    ground_truth = Ytest;
    [Xtrain,Ytrain,Xtest,Ytest] = initial(Xtrain, Ytrain, 'f', Xtest, Ytest); % standardization
elseif preprocess == 2
    % normalise or standardize only features
    Xtrain = (Xtrain - min(X)) ./ (max(X) - min(X));
    Xtest = (Xtest - min(X)) ./ (max(X) - min(X));
    Ytrain = (Ytrain - min(Y)) ./ (max(Y) - min(Y));
    Ytest = (Ytest - min(Y)) ./ (max(Y) - min(Y));
end
%% Committee Network
type_svm = 0;
nb_partitioning = 2; % times to divide training set in 2 sets
partitions = partition(Xtrain, Ytrain, nb_partitioning)';
P = length(partitions);
N = length(Ytrain);
Nt = length(Ytest);
costs = zeros(P);
%models = zeros(P);
kernel = 'RBF_kernel';
train_sim = zeros(N,P);
test_sim = zeros(Nt,P);
for idx = 1:P % set will hold a 2x1 cell, X above, Y below
    set = partitions(:,idx);
    if type_svm == 0 % ls-svm
        Xtr = cell2mat(set(1));
        Ytr = cell2mat(set(2));
        [gamma,params,cost] = tunelssvm({Xtr, Ytr, 'f', [], [], kernel}, 'simplex', 'crossvalidatelssvm', {10, 'mse'});
        costs(idx) = cost;
        parameters = {Xtr, Ytr, 'f', gamma, params, kernel};
        [alpha,b] = trainlssvm(parameters); % {alpha,b} = ...
        %models(idx) = {alpha b};
        train_sim(:,idx) = simlssvm(parameters, {alpha,b}, Xtrain);
        test_sim(:,idx) = simlssvm(parameters, {alpha,b}, Xtest);
    else % classical svm
        
    end
end
%% Linear approach
% Covariance matrix
C = zeros(P,P);
for i = 1:P
    sim_i = train_sim(:,i);
    for j = i:P
        sim_j = train_sim(:,j);
        v = mean(sum((sim_i - Ytrain) .* (sim_j - Ytrain)));
        C(i,j) = v;
        C(j,i) = v;
    end
end
vec = ones(P,1);
%beta = (inv(C)*vec) / (vec'*inv(C)*vec);
beta = ((C \ vec) / (vec' / C * vec))';
%% Measure error
train_answer = sum(beta .* train_sim(:,:), 2);
test_answer = sum(beta .* test_sim(:,:), 2);
train_error = mae(train_answer - Ytrain);
test_error = mae(test_answer - Ytest);
train_error_mse = mse(train_answer - Ytrain);
test_error_mse = mse(test_answer - Ytest);
fprintf('Error on training set = %.5f\n', train_error);
fprintf('Error on test set = %.5f\n', test_error);
fprintf('Error on training set (mse) = %.5f\n', train_error_mse);
fprintf('Error on test set (mse) = %.5f\n', test_error_mse);
if preprocess == 1
    %mae(ground_truth - test_answer * std(Y) + mean(Y))
end
%% Visualize
if sinc_demo
    figure
    hold on
    plot(Xtest,test_answer,'r.', 'Color', [0.6740, 0.2, 0.1880]);
    plot(Xtest,Ytest,'b.', 'Color', [0, 0.4470, 0.7410]);
    if preprocess == 1
        plot((X-mean(X))/std(X),(sinc(X)-mean(Y))/std(Y),'g.', 'Color', [0.4470, 0.7410, 0]);
    elseif preprocess == 2
        plot((X-min(X))/(max(X)-min(X)),(sinc(X)-min(Y))/(max(Y)-min(Y)),'g.', 'Color', [0.4470, 0.7410, 0]);
    else
        plot(Xtest,sinc(Xtest),'g.', 'Color', [0.4470, 0.7410, 0]);
    end
    legend('committee','dataset','dataset-noise')
end
%% MLP approach
c = cvpartition(Ytrain, 'HoldOut', 0.5, 'Stratify', true);
XtrainMLP = train_sim(c.training,:)';
YtrainMLP = Ytrain(c.training)';
XvalMLP = train_sim(c.test,:)';
YvalMLP = Ytrain(c.test)';
layers = [
    sequenceInputLayer(4) %, 'Normalization', 'zerocenter'
    fullyConnectedLayer(1)
    regressionLayer
    %maeRegressionLayer()
];
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 1200, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XvalMLP,YvalMLP}, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
net = trainNetwork(XtrainMLP, YtrainMLP, layers, options);
%% MLP prediction
YtestMLP = predict(net, test_sim');
fprintf('MLP MAE for test set = %.5f\n', mse(YtestMLP-Ytest'));
if preprocess == 1
    % mae(YtestMLP' * std(Y) + mean(Y) - ground_truth)
end
%% Functions
function partitions = partition(X, Y, i)
    c = cvpartition(Y, 'HoldOut', 0.5, 'Stratify', true);
    if i <= 1
        partitions = vertcat({X(c.training,:),Y(c.training)},{X(c.test,:),Y(c.test)});
    else
        partitions = vertcat(...
            partition(X(c.training,:),Y(c.training),i-1),...
            partition(X(c.test,:),Y(c.test),i-1));
    end
end