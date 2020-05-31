%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
addpath(genpath('svm'))
addpath(genpath('smote'))
close all
%% Load dataset
load('diabetes')
%X = [total ; testset];
%Y = [labels_total ; labels_test];

Xtrain = total;
Ytrain = labels_total;
Xtest = testset;
Ytest = labels_test;
% %% Do KNN
% for neighbours = 1:10
%     nns = knnsearch(Xtrain, Xtest, 'K', neighbours);
%     errors = sum(Ytest ~= Ytrain(mode(nns,2)));
%     accuracy = 1 - errors/length(Ytest);
%     fprintf('Accuraccy of KNN (K = %d) = %.5f\n', neighbours, accuracy);
% end
% %% Random Forest Classifier
%[x_train, y_train] = SMOTE(Xtrain, Ytrain);
nTrees = 300;
rf = TreeBagger(nTrees, x_train, y_train, 'Method', 'classification'); %, 'NumPredictorsToSample', 1
Ypred = str2double(rf.predict(Xtest));
accuracy = sum(Ytest == Ypred) / length(Ytest);
fprintf('Accuraccy of RFC = %.5f\n', accuracy);
%% LS-SVM
kernel = 'RBF_kernel';
aucs = [];
costs = [];
tune_params = {Xtrain, Ytrain, 'c', 0.1, [], kernel};
[gam,params,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {5, 'misclass'});
fprintf('Tuning results (g,t,c) = (%.3f,%.3f,%.3f,%.3f)\n', gam, params, cost)
parameters = {Xtrain, Ytrain, 'c', gam, params, kernel};
% Training
[alpha,b] = trainlssvm(parameters);
fprintf('Number of support vectors = %d\n', length(alpha));
% Testing
[test_classes, Ylatent] = simlssvm(parameters, {alpha,b}, Xtest);
miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
fprintf('Missclassification rate : %.5f\n', miss_rate)
% ROC
[auc,~,thresholds,fpr,tpr] = roc(Ylatent, Ytest);
fprintf('ROC_AUC : %.5f\n', auc)
%%
rates = zeros(length(Ytest),2);
i = 0;
for t = thresholds'
    [~, Ylatent] = simlssvm(parameters, {alpha,b}, Xtest);
    test_classes = (Ylatent > t) - (Ylatent <= t);
    miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
    i = i+1;
    rates(i,1) = miss_rate;
    rates(i,2) = t;
end
min(rates) %#ok