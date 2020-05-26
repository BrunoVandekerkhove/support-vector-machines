%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
addpath(genpath('svm'))
close all
%% 2.1a Ripley dataset (LSSVM)
load('ripley')
visualise(Xtrain, Ytrain)%, Xtest, Ytest)
export_pdf(gcf, 'classification/ripley')
for kernel = ["RBF_kernel"]
    % Tuning
    tune_params = {Xtrain, Ytrain, 'c', 0.1, [], kernel};
    [gam,sig2,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
    fprintf('Tuning results (g,s,c) = (%.3f,%.3f,%.3f)\n', gam, sig2, cost)
    % Training
    parameters = {Xtrain, Ytrain, 'c', gam, sig2, kernel};
    [alpha,b] = trainlssvm(parameters);
    fprintf('Number of support vectors = %d\n', length(alpha));
    % Testing
    [test_classes, Ylatent] = simlssvm(parameters, {alpha,b}, Xtest);
    miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
    fprintf('Missclassification rate : %.5f\n', miss_rate)
    % Visualisation
    plotlssvm(parameters, {alpha,b});
    %export_pdf(gcf, sprintf('classification/ripley_%s', kernel))
    % ROC
    [auc,~,thresholds,fpr,tpr] = roc(Ylatent, Ytest);
    %export_pdf(gcf, sprintf('classification/ripley_roc_%s', kernel))
    fprintf('ROC_AUC : %.5f\n', auc)
end
%% 2.1b Ripley dataset (SVM)
% Note that the kernel parameters are globals (p1 and p2 respectively)
% So to set them you can do so anywhere
% eg p1 = ? to set bandwidth of RBF kernel
load('ripley')
% uiclass
global p1 p2;
for kernel = ["sigmoid", "spline", "bspline","linear", "poly", "rbf"]%, "fourier", "erfb", "anova"]
    gamma = 1.0;
    switch(lower(kernel))
        case "poly"
            gamma = 10.0;
            p1 = 5;
        case "rbf"
            gamma = 0.5;
            p1 = 0.15;
        case "sigmoid"
            p1 = 1.0;
            p2 = 1.0;
        case "spline"
            p1 = 1.0;
            p2 = 1.0;
        case "bspline"
            p1 = 1.0;
            p2 = 1.0;
        otherwise
            p1 = 1.0;
            p2 = 1.0;
    end
    [nsv,alpha,bias] = svc(Xtrain, Ytrain, kernel, gamma);
    svcplot(Xtrain, Ytrain, kernel, alpha, bias);
    %output = svcoutput(Xtrain, Ytrain, Xtest, kernel, alpha, bias);
    % if you look at the code it's the misclassification error (#misclassed) when using
    %   sgn function to predict
    error = svcerror(Xtrain, Ytrain, Xtest, Ytest, kernel, alpha, bias);
    fprintf('Error for kernel %s = %.5f\n', kernel, error/length(Xtest));
    %export_pdf(gcf, sprintf('classification/ripley_svm_%s', kernel))
    waitforbuttonpress;
end
%% 2.2a Breast Cancer dataset (visualization)
load('breast')
Xtrain = trainset;
Ytrain = labels_train;
Xtest = testset;
Ytest = labels_test;
%visualise(trainset, labels_train, 1);
%export_pdf(gcf, 'classification/wisconsin_pca')
%% 2.2b Breast Cancer dataset (basic data analysis)
for i = 1:4
   figure
   hold on
   histogram(Xtrain(Ytrain == 1, i*3-2))
   histogram(Xtrain(Ytrain == -1, i*3-2))
   %legend('Benign', 'Malignant')
   hold off
   %export_pdf(gcf, sprintf('classification/breast_hist_%d', i));
end
close all;
%% 2.2c Breast Cancer dataset (LS-SVM)
for kernel = ["lin_kernel", "poly_kernel", "RBF_kernel"]
    aucs = [];
    for i = 1:10
        % Tuning
        tune_params = {Xtrain, Ytrain, 'c', 0.1, [], kernel};
        if kernel == "poly_kernel"
            [gam,params,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
            fprintf('Tuning results (g,t,d,c) = (%.3f,%.3f,%.3f,%.3f)\n', gam, params(1), params(2), cost)
            parameters = {Xtrain, Ytrain, 'c', gam, params, kernel};
        else
            [gam,sig2,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
            fprintf('Tuning results (g,s,c) = (%.3f,%.3f,%.3f)\n', gam, sig2, cost)
            parameters = {Xtrain, Ytrain, 'c', gam, sig2, kernel};
        end
        % Training
        %parameters = {Xtrain, Ytrain, 'c', gam, sig2, kernel};
        [alpha,b] = trainlssvm(parameters);
        fprintf('Number of support vectors = %d\n', length(alpha));
        % Testing
        [test_classes, Ylatent] = simlssvm(parameters, {alpha,b}, Xtest);
        miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
        fprintf('Missclassification rate : %.5f\n', miss_rate)
        % ROC
        [auc,~,thresholds,fpr,tpr] = roc(Ylatent, Ytest);
        %export_pdf(gcf, sprintf('classification/ripley_roc_%s', kernel))
        fprintf('ROC_AUC : %.5f\n', auc)
        aucs = [auc aucs]; %#ok
    end   
    fprintf('Mean AUC for kernel %s = %.3f (std = %.3f)', kernel, mean(aucs), std(aucs));
end
%% 2.2d Breast Cancer dataset (ARD)
%tune_params = {Xtrain, Ytrain, 'c', 0.1, [], 'RBF_kernel'};
%[gam,sig2,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
%fprintf('Tuning results (g,s,c) = (%.3f,%.3f,%.3f)\n', gam, sig2, cost)
%[selected, ranking] = bay_lssvmARD({Xtrain, Ytrain, 'c', gam, sig2}) %#ok
%save('breast_ard', 'selected', 'ranking') % save intermediate results
load('breast_ard') % load intermediate results
% Train with relevant inputs
Xard = Xtrain(:,selected); % relevant inputs of training set
Xtard = Xtest(:,selected); % relevant inputs of test set
aucs = [];
costs = [];
for i = 1:10
    tune_params = {Xard, Ytrain, 'c', 0.1, [], 'lin_kernel'};
    [gam,sig2,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
    fprintf('Tuning results (g,s,c) = (%.3f,%.3f,%.3f)\n', gam, sig2, cost)
    parameters = {Xard, Ytrain, 'c', gam, sig2, 'lin_kernel'};
    [alpha,b] = trainlssvm(parameters);
    [test_classes, Ylatent] = simlssvm(parameters, {alpha,b}, Xtard);
    miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
    fprintf('Missclassification rate : %.5f\n', miss_rate);
    [auc] = roc(Ylatent, Ytest);
    fprintf('AUC = %.5f\n', auc);
    aucs = [aucs auc] %#ok
    costs = [costs cost] %#ok
end
fprintf('Mean AUC / cost = %.5f (%.5f) %.5f (%.5f)', mean(aucs), std(aucs), mean(costs), std(costs))
%% 2.2e Breast Cancer dataset (PCA)
pca_test(Xtrain, Ytrain, Xtest, Ytest, [3,10,20])
%% 2.3a Diabetes dataset (visualization)
load('diabetes')
Xtrain = total;
Ytrain = labels_total;
Xtest = testset;
Ytest = labels_test;
%visualise(trainset, labels_train, 1)
%export_pdf(gcf, 'classification/diabetes_pca')
%% 2.3b Diabetes dataset (basic data analysis)
for i = 1:size(Xtrain,2)
   figure
   hold on
   histogram(Xtrain(Ytrain == 1, i))
   histogram(Xtrain(Ytrain == -1, i))
   %legend('Benign', 'Malignant')
   hold off
   export_pdf(gcf, sprintf('classification/diabetes_hist_%d', i));
end
close all;
%% 2.3c Diabetes dataset (LS-SVM)
for kernel = ["lin_kernel", "poly_kernel", "RBF_kernel"]
    aucs = [];
    costs = [];
    for i = 1:3
        % Tuning
        tune_params = {Xtrain, Ytrain, 'c', 0.1, [], kernel};
        if kernel == "poly_kernel"
            [gam,params,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
            fprintf('Tuning results (g,t,d,c) = (%.3f,%.3f,%.3f,%.3f)\n', gam, params(1), params(2), cost)
            parameters = {Xtrain, Ytrain, 'c', gam, params, kernel};
        else
            [gam,sig2,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
            fprintf('Tuning results (g,s,c) = (%.3f,%.3f,%.3f)\n', gam, sig2, cost)
            parameters = {Xtrain, Ytrain, 'c', gam, sig2, kernel};
        end
        % Training
        %parameters = {Xtrain, Ytrain, 'c', gam, sig2, kernel};
        [alpha,b] = trainlssvm(parameters);
        fprintf('Number of support vectors = %d\n', length(alpha));
        % Testing
        [test_classes, Ylatent] = simlssvm(parameters, {alpha,b}, Xtest);
        miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
        costs = [costs miss_rate]; %#ok
        fprintf('Missclassification rate : %.5f\n', miss_rate)
        % ROC
        [auc,~,thresholds,fpr,tpr] = roc(Ylatent, Ytest);
        %export_pdf(gcf, sprintf('classification/ripley_roc_%s', kernel))
        fprintf('ROC_AUC : %.5f\n', auc)
        aucs = [auc aucs]; %#ok
    end   
    close all
    fprintf('Mean cost for kernel %s = %.3f (std = %.3f)\n', kernel, mean(costs), std(costs));
    fprintf('Mean AUC for kernel %s = %.3f (std = %.3f)\n', kernel, mean(aucs), std(aucs));
end
%% 2.3d Diabetes dataset (ARD)
tune_params = {Xtrain, Ytrain, 'c', 0.1, [], 'RBF_kernel'};
[gam,sig2,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
fprintf('Tuning results (g,s,c) = (%.3f,%.3f,%.3f)\n', gam, sig2, cost)
[selected, ranking] = bay_lssvmARD({Xtrain, Ytrain, 'c', gam, sig2}) %#ok
save('diabetes_ard', 'selected', 'ranking') % save intermediate results
%% 2.3e Diabetes dataset (PCA)
pca_test(Xtrain, Ytrain, Xtest, Ytest, [2,3,5])
%% Functions
function visualise(X, y, use_pca)
    if nargin < 3; use_pca = 0; end
    dim = size(X,2);
    if dim > 2
        if use_pca
            [~, score] = pca(X.');
            X = score(:, 1:2);
        else
            X = tsne(X);
        end
    end
    figure
    hold on
    plot(X(y>0,1), X(y>0,2), 'ko','MarkerFaceColor','b');
    plot(X(y<0,1), X(y<0,2), 'ko','MarkerFaceColor','r');
    hold off
end
function export_pdf(h, output_name)
%EXPORT_PDF Exports the given figure to a pdf file.
    set(h, 'PaperUnits','centimeters');
    set(h, 'Units','centimeters');
    pos=get(h,'Position');
    set(h, 'PaperSize', [pos(3) pos(4)]);
    set(h, 'PaperPositionMode', 'manual');
    set(h, 'PaperPosition',[0 0 pos(3) pos(4)]);
    print('-dpdf', strcat('figures/', output_name));
end
function pca_test(Xtrain, Ytrain, Xtest, Ytest, components)
    mean_train = mean(Xtrain);
    [~,eigvec] = pca(Xtrain-mean_train, max(components));
    for n_h = components
        % do pca
        proj = eigvec(:,1:n_h);
        % project train/test data
        train_proj = (Xtrain - mean_train) * proj;
        test_proj = (Xtest - mean_train) * proj;
        costs = [];
        aucs = [];
        % tune/train on training data
        for i = 1:10
           tune_params = {train_proj, Ytrain, 'c', 0.1, [], 'RBF_kernel'};
            [gam,sig2,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
            fprintf('Tuning results (g,s,c) = (%.3f,%.3f,%.3f)\n', gam, sig2, cost)
            parameters = {train_proj, Ytrain, 'c', gam, sig2, 'RBF_kernel'};
            [alpha,b] = trainlssvm(parameters);
            [test_classes, Ylatent] = simlssvm(parameters, {alpha,b}, test_proj);
            miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
            fprintf('Missclassification rate : %.5f\n', miss_rate);
            [auc] = roc(Ylatent, Ytest);
            fprintf('AUC = %.5f\n', auc);
            aucs = [aucs auc] %#ok
            costs = [costs cost] %#ok 
        end
        close all
        fprintf('Mean AUC / cost = %.5f (%.5f) %.5f (%.5f)', mean(aucs), std(aucs), mean(costs), std(costs))
    end
end