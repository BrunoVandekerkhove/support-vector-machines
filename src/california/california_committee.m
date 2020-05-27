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
data = data(1:1600,:);
sinc_demo = 0;
if sinc_demo
    X = (-3:0.0025:3)'; 
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
% Normalize (?)
%Xtrain = (Xtrain - min(Xtrain)) ./ (max(Xtrain) - min(Xtrain));
%Xtest = (Xtest - min(Xtest)) ./ (max(Xtest) - min(Xtest));
% Standardize (?)
%[Xtrain,Ytrain,Xtest,Ytest] = initial(Xtrain, Ytrain, 'f', Xtest, Ytest); % standardization
%[Xtrain,~,Xtest,~] = initial(Xtrain, Ytrain, 'f', Xtest, Ytest); % standardization (only features)
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
%%
train_answer = sum(beta .* train_sim(:,:), 2);
test_answer = sum(beta .* test_sim(:,:), 2);
train_error = mae(train_answer - Ytrain);
test_error = mae(test_answer - Ytest);
fprintf('Error on training set = %.5f\n', train_error);
fprintf('Error on test set = %.5f\n', test_error);
%% Visualize
if sinc_demo
    figure
    hold on
    plot(Xtest,test_answer,'r.', 'Color', [0.6740, 0.2, 0.1880]);
    plot(Xtest,Ytest,'b.', 'Color', [0, 0.4470, 0.7410]);
    plot(Xtest,sinc(Xtest),'g.', 'Color', [0.4470, 0.7410, 0]);
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
fprintf('MLP MAE for test set = %.5f\n', mae(YtestMLP-Ytest'));
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