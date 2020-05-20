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
for kernel = ["lin_kernel", "RBF_kernel"]
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
%% 2.2 Breast Cancer dataset
load('breast')
Xtrain = trainset;
Ytrain = labels_train;
Xtest = testset;
Ytest = labels_test;
%visualise(trainset, labels_train, 1)
%export_pdf(gcf, 'classification/wisconsin')
for kernel = ["lin_kernel", "poly_kernel", "RBF_kernel"]
    % Tuning
    tune_params = {Xtrain, Ytrain, 'c', 0.1, [], kernel};
    if kernel == "poly_kernel"
        [gam,tau,degree,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
        fprintf('Tuning results (g,t,d,c) = (%.3f,%.3f,%.3f,%.3f)\n', gam, tau, degree, cost)
        parameters = {Xtrain, Ytrain, 'c', gam, [tau,degree], kernel};
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
end
%% 2.3 Diabetes dataset
load('diabetes')
visualise(trainset, labels_train)
%export_pdf(gcf, 'classification/diabetes')
%% Functions
function visualise(X, y, use_pca)
    if nargin < 3; use_pca = 0; end
    dim = size(X,2);
    if dim > 2
        if use_pca
            [~, score] = pca(X.');
            X = score(:, 1:2);
            size(X)
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