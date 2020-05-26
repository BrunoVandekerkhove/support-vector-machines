% Read data
dataset = 2;
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
end
%addpath('../LSSVMlab')

% Set up data
X = data(:,1:end-1);
Y = data(:,end);
if dataset == 2 && 1 % set to 1 to do multiclass classification with test set
    % Build train / val / test (random split, stratified)
    c = cvpartition(Y, 'HoldOut', 0.2, 'Stratify', true);
    Xtr = X(c.training,:);
    Ytr = Y(c.training);
    Xtest = X(c.test,:);
    Ytest = Y(c.test);
    sum(Ytest==1)/length(Ytest)
else
    Xtr = X;
    Ytr = Y;
    Xtest = [];
    Ytest = [];
end

global repetitions
repetitions = 2;

% Parameter for input space selection
% Please type >> help fsoperations; to get more information  
k = 6;
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

% Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

% Perform FS-LSVM
[e,s,t] = fslssvm(Xtr, Ytr, k, function_type, kernel_type, global_opt, user_process, window, Xtest, Ytest);
