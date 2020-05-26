%% Initialisation
clc
clear
addpath(genpath('./figures'))
addpath(genpath('./lssvm'))
addpath(genpath('./data'))
close all
%% Read data
data = load('shuttle.dat', '-ascii'); 
data = data(1:10000,:);
%% Set up data
classes_to_keep = 1:5;
indices_to_keep = ismember(data(:,end), classes_to_keep);
use_testset = 1;
X = data(indices_to_keep,1:end-1);
Y = data(indices_to_keep,end);
if use_testset % set to 1 to do multiclass classification with test set
    % Build train / val / test (random split, stratified)
    c = cvpartition(Y, 'HoldOut', 0.25, 'Stratify', true);
    Xtrain = X(c.training,:);
    Ytrain = Y(c.training);
    Xtest = X(c.test,:);
    Ytest = Y(c.test);
    fprintf('Class imbalance : %.5f\n', sum(Ytest==1) / length(Ytest));
    for cl = 1:7
        fprintf('# (class = %d) : %d\n', cl, sum(Ytest==cl));
    end
else
    Xtrain = X;
    Ytrain = Y;
    Xtest = [];
    Ytest = [];
end
%% Set estimation parameters
repetitions = 1;
k = 2;
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
user_processes = ["SV_L0_norm"]; %"FS-LSSVM"
class_order = [1,4,5,3];
%% Perform LS-SVM
bt = cputime;
windowrange = [10,15,20];
process_matrix_err = [];
process_matrix_sv = [];
process_matrix_time = [];
%% Train model
ups = length(user_processes);
e = zeros(ups, repetitions);
s = zeros(ups, repetitions);
t = zeros(ups, repetitions);
for up_sidx = 1:ups
    user_process = user_processes(up_sidx);
    for rep = 1:repetitions
        X = Xtrain; Y = Ytrain; Xt = Xtest; Yt = Ytest;
        totalerrors = 0;
        totaltime = 0;
        totalspvcs = 0;
        for pick = class_order
            [spvc, time, result] = fixed_size_apply(pick, ...
                X, Y, Xt, Yt, ...
                kernel_type, k, user_process);
            if pick == class_order(end)
                next_indices = find(result < 0); % the indices of 'not this class'
                totalerrors = totalerrors + ...
                    sum(Yt ~= pick & result == 1) + ...
                    sum(ismember(Yt(next_indices), class_order));
            else
                totalerrors = totalerrors + sum(Yt ~= pick & result == 1);
                next_indices = find(result < 0); % the indices of 'not this class'
                % X & Y are for training / validation so take the proper
                % classes (the remaining ones)
                X = X(Y ~= pick,:);
                Y = Y(Y ~= pick,:);
                % Since results of testing filtered out some samples they
                % need to be taken out even when predictions were wrong
                Xt = Xt(next_indices,:);
                Yt = Yt(next_indices,:);
            end
            totaltime = totaltime + time;
            totalspvcs = totalspvcs + spvc;
        end
        e(up_sidx, rep) = totalerrors;% / length(Ytest);
        s(up_sidx, rep) = totalspvcs;
        t(up_sidx, rep) = totaltime;
    end
end
%% Visualize results
% figure;
% boxplot(e, 'Label', user_process);
% ylabel('Error estimate');
% title('Error Comparison for different approaches (user processes)');
% figure;
% boxplot(s, 'Label', user_process);
% ylabel('SV estimate');
% title('Number of SV for different approaches (user processes)');
% figure;
% boxplot(t, 'Label', user_process);
% ylabel('Time estimate');
% title('Comparison for time taken by different approaches (user processes)');
%% Functions
function [spvc, time, testYh] = fixed_size_apply(selected_class, ...
    Xtrain, Ytrain, Xtest, Ytest, ...
    kernel_type, k, user_process)
    global_opt = 'csa';
    folds = 5;
    % prepare data
    Y = (Ytrain == selected_class) - (Ytrain ~= selected_class);
    Yt = (Ytest == selected_class) - (Ytest ~= selected_class);
    % renyi
    [svX, svY, subset, renyitime] = renyi(Xtrain, Y, kernel_type, k);
    % tune
    if user_process == "SV_L0_norm"
        [gam,sig] = tunelssvm({svX, svY, 'c', [], [], kernel_type, global_opt}, 'simplex', 'crossvalidatelssvm', {folds,'misclass'});
    else

        [gam,sig] = tunefslssvm({Xtrain, Y, 'c', [], [], kernel_type, global_opt}, svX, folds, 'misclass', 'simplex');
    end
    traintime = cputime;
    [error, newsvX, newsvY, testYh] = testmodsparseoperations(Xtrain, Y, Xtest, Yt, svX, svY, subset, sig, gam, kernel_type, 'c', user_process, []);
    time = cputime - traintime + renyitime;
    fprintf('Error (distinguishing class %d) :( = %.5f\n', selected_class, error);
    %errors = sum((testYh == 1) & (Yt ~= 1)); % no percentage (easier to calculate total)
    spvc = (size(newsvX,1) + length(newsvY))/2;
end
function [svX, svY, subset, basetime] = renyi(Xtrainf, Ytrainf, kernel_type, k)
    N = length(Xtrainf);
    dim = size(Xtrainf,2);
    represent_points = ceil(k*sqrt(N)); % number of prototype vectors
    sig2 = (std(Xtrainf)*(N^(-1/(dim+4)))).^2; % renyi entropy density estimation
    renyit = cputime;
    subset = entropysubset(Xtrainf,represent_points,kernel_type,sig2,[]);
    basetime = cputime - renyit;
    subset = subset';
    svX = Xtrainf(subset,:);
    svY = Ytrainf(subset,:);
end