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
Xtest = [];
Ytest = [];

% Build train / val / test (random split, stratified)
% c = cvpartition(Y, 'HoldOut', 0.5, 'Stratify', true);
% Xother = X(c.test);
% Yother = Y(c.test);
% c2 = cvpartition(Y(c.test), 'Holdout', 0.5, 'Stratify', true);
% Xtr = X(c.train);
% Xval = Xother(c2.train);
% Xtest = Xother(c2.test);
% Ytr = Y(c.train);
% Yval = Yother(c2.train);
% Ytest = Yother(c2.test);

% Parameter for input space selection
% Please type >> help fsoperations; to get more information  
k = 2;
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

% Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

% Perform FS-LSVM
[e,s,t] = fslssvm(X, Y, k, function_type, kernel_type, global_opt, user_process, window, Xtest, Ytest);
