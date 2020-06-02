clear
close all
clc
%%
% Read data
dataset = 3;
switch dataset
    case 1
        data = load('breast_cancer_wisconsin_data.mat', '-ascii'); 
        function_type = 'c';
    case 2
        data = load('shuttle.dat', '-ascii'); 
        function_type = 'c';  
        data = data(1:700,:);
    case 3
        data = load('california.dat', '-ascii'); 
        function_type = 'f';
        %data = data(1:160,:);
end
%addpath('../LSSVMlab')

% Set up data
X = data(:,1:end-1);
Y = data(:,end);
if dataset == 2 && 1 % set to 1 to do multiclass classification with test set
    % Build train / val / test (random split, stratified)
    c = cvpartition(Y, 'HoldOut', 0.25, 'Stratify', true);
    Xtr = X(c.training,:);
    Ytr = Y(c.training);
    Xtest = X(c.test,:);
    Ytest = Y(c.test);
    % sum(Ytest==1)/length(Ytest) % class imbalance
elseif dataset == 3
    filter = Y < 500000; % to take away highly priced houses
    X = X(filter,:);
    Y = Y(filter);
    trainsize = ceil(3*length(Y)/4);
    Xtrain = X(1:trainsize,:);
    Ytrain = Y(1:trainsize,:);
    Xtest = X(trainsize+1:end,:);
    Ytest = Y(trainsize+1:end,:);   
else
    Xtrain = X;
    Ytrain = Y;
    Xtest = [];
    Ytest = [];
end

global repetitions
repetitions = 1;

% Parameter for input space selection
% Please type >> help fsoperations; to get more information  
k = 4;
kernel_type = 'poly_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

% Process to be performed
user_process={'FS-LSSVM'};%, 'SV_L0_norm'
window = [15,20,25];

% Perform FS-LSVM
[e,s,t] = fslssvm(Xtrain, Ytrain, k, function_type, kernel_type, global_opt, user_process, window, Xtest, Ytest);


%