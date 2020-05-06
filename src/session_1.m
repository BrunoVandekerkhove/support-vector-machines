%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
close all
%% 1.1 A simple example : two Gaussians
X1 = randn(50,2) + 1; % Center at (1,1)
X2 = randn(51,2) - 1; % Center at (-1,-1)
Y1 = ones(50,1); % Positive class labels
Y2 = -ones(51,1); % Negative class labels
figure;
hold on;
plot(X1(:,1), X1(:,2), 'ro','MarkerFaceColor','r');
plot(X2(:,1), X2(:,2), 'bo','MarkerFaceColor','b');
fplot(@(x) -x, '-')
xlabel('X1')
ylabel('X2')
legend('Positive Class', 'Negative Class', 'Decision Boundary')
hold off;
export_pdf(gcf, 'twogaussians')
%% 1.3a LS-SVM classifier (polynomial)
% democlass
load('data/iris')
t = 1; % Polynomial coefficient
for d = 1:4 % Degree
    for gamma = 1.0 % Regularization parameter
        parameters = {Xtrain, Ytrain, 'classification', gamma, [t,d], 'poly_kernel'};
        [alpha,b] = trainlssvm(parameters);
        test_classes = simlssvm(parameters, {alpha,b}, Xtest);
        miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
        fprintf('Degree %d, missclassification rate : %.2f\n', d, miss_rate)
        plotlssvm(parameters, {alpha,b});
        waitforbuttonpress
        % export_pdf(gcf, sprintf('iris/%d_%d', d, gamma*10))
    end
end
%% 1.3b LS-SVM classifier (RBF)
load('data/iris')
samples = 10.^(-3:0.05:3)
N = length(samples); % Number of values to test
% Prepare figure
figure
hold on
% Experiment with sigma
miss_rates_s = ones(1,N); % Misclassification rates
for gamma = 1.0 % Regularization parameter (fixed)
    for i = 1:N
        parameters = {Xtrain, Ytrain, 'classification', gamma, samples(i), 'RBF_kernel'};
        [alpha,b] = trainlssvm(parameters);
        test_classes = simlssvm(parameters, {alpha,b}, Xtest);
        miss_rates_s(i) = sum(test_classes ~= Ytest) / length(Ytest);
    end
    p = plot(samples, miss_rates_s, 'Linewidth', 3);
    set(p, 'Color', [0 0.4470 0.7410]);
end
% Experiment with gamma
miss_rates_g = ones(1,N); % Misclassification rates
for sigma = 1.0 % Bandwith (fixed)
    for i = 1:N
        parameters = {Xtrain, Ytrain, 'classification', samples(i), sigma, 'RBF_kernel'};
        [alpha,b] = trainlssvm(parameters);
        test_classes = simlssvm(parameters, {alpha,b}, Xtest);
        miss_rates_g(i) = sum(test_classes ~= Ytest) / length(Ytest);
    end
    p = plot(samples, miss_rates_g, 'Linewidth', 3);
    set(p, 'Color', [0.9290 0.6940 0.1250]);
end
% Visualize
xlabel('\sigma^2 / \gamma')
ylabel('error')
legend('\gamma=1.0, variable \sigma^2', '\sigma^2=1.0, variable \gamma')
set(gca,'xscale','log')
hold off
export_pdf(gcf, 'iris/tuning')
%% 1.3c LS-SVM classifier (compare to sample script)
load('data/iris')
iris_sample_script
close all
clc
%% 1.3d LS-SVM classifier (tuning parameters using validation)
load('data/iris')
values = exp(linspace(0,log(1E6),21)) * 1E-3;
values_interpolated = exp(linspace(0,log(1E6),200)) * 1E-3; %10.^(-3:0.5:3);
N = length(values);
results = ones(N,N);
strategies = {
    @(p) rsplitvalidate(p, 0.80, 'misclass')
    @(p) crossvalidate(p, 10, 'misclass') 
    @(p) leaveoneout(p, 'misclass')
};
figure
xlabel('gamma')
ylabel('sigma')
for strategy = 1:length(strategies)
    for i = 1:N
        for j = 1:N
            parameters = {Xtrain, Ytrain, 'c', values(j), values(N-i+1), 'RBF_kernel'}; % Note values(N-i+1)
            results(i,j) = strategies{1}(parameters);
        end
    end
    % colormap(flipud(hot))
    colormap(flipud(parula))
    % map = interp2(results, 7, 'nearest');
    [X,Y] = meshgrid(values, values);
    [Xq,Yq] = meshgrid(values_interpolated, values_interpolated);
    Vq = interp2(X, Y, results, Xq, Yq);
    % contourf(Xq,Yq,Vq); % Interpolated version
    contourf(values, values, results);
    set(gca, 'XScale', 'log','YScale', 'log');
    xlabel('\sigma^2')
    ylabel('\gamma')
    colorbar
    fprintf('Press any key ...')
    export_pdf(gcf, sprintf('iris/validation_%d', strategy))
    waitforbuttonpress
end
%% 1.3e LS-SVM classifier (automated parameter tuning)
load('data/iris')
display_runtime = 1;
% Simplex (Nelder-Mead)
parameters = {Xtrain, Ytrain, 'c', [], [], 'RBF_kernel'};
if display_runtime tic; end %#ok
N = 10; % Number of runs
gsimp = ones(1,N);
ssimp = ones(1,N);
csimp = ones(1,N);
for i = 1:N
    [gsimp(i),ssimp(i),csimp(i)] = tunelssvm(parameters, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
end
fprintf('Average (gamma,sigma2,cost) = (%f,%f,%f)\n', mean(gsimp), mean(ssimp), mean(csimp))
fprintf('Std (gamma,sigma2,cost) = (%f,%f,%f)\n', std(gsimp), std(ssimp), std(csimp))
if display_runtime toc; end %#ok
% Grid Search
if display_runtime tic; end %#ok
ggrid = ones(1,N);
sgrid = ones(1,N);
cgrid = ones(1,N);
for i = 1:N
    [ggrid(i),sgrid(i),cgrid(i)] = tunelssvm(parameters, 'gridsearch', 'crossvalidatelssvm', {10, 'misclass'});
end
fprintf('Average (gamma,sigma2,cost) = (%f,%f,%f)\n', mean(ggrid), mean(sgrid), mean(cgrid))
fprintf('Std (gamma,sigma2,cost) = (%f,%f,%f)\n', std(ggrid), std(sgrid), std(cgrid))
if display_runtime toc; end %#ok
%% Histograms for automated tuning
histogram(ggrid, 10)
export_pdf(gcf, 'iris/histogram_gamma')
waitforbuttonpress
histogram(sgrid, 10)
export_pdf(gcf, 'iris/histogram_sigma2')
waitforbuttonpress
histogram(cgrid, 10)
export_pdf(gcf, 'iris/histogram_cost')
%% 1.3f LS-SVM classifier (using ROC curves)
load('data/iris')
gamma = 0.037622;
sigma = 0.559597;
parameters = {Xtrain, Ytrain, 'c', gamma, sigma, 'RBF_kernel'};
[alpha, b] = trainlssvm(parameters);
[~, Ylatent] = simlssvm(parameters, {alpha, b}, Xtest);
roc(Ylatent, Ytest);
export_pdf(gcf, 'iris/roc')
%% 1.3g LS-SVM classifier (Bayesian framework)
load('data/iris')
% for i = 1:length(gsimp) % Requires 1.3e
%     parameters = {Xtrain, Ytrain, 'c', gsimp(i), ssimp(i)};
%     bay_modoutClass(parameters, 'figure');
%     colorbar
%     export_pdf(gcf, sprintf('iris/bayesian/bayesian_%d', i))
%     waitforbuttonpress
% end
gamma = 0.037622;
sigma = 0.559597;
parameters = {Xtrain, Ytrain, 'c', gamma, sigma, 'RBF_kernel'};
bay_modoutClass(parameters, 'figure');
colorbar
export_pdf(gcf, 'iris/bayesian/bayesian_probabilities')
%% 1.3h LS-SVM classifier (misc)
% This visualises the classifiers for the gamma/sigma2 values found through
% automated tuning.
load('data/iris')
for i = 1:length(gsimp) % Requires 1.3e
    parameters = {Xtrain, Ytrain, 'classification', gsimp(i), ssimp(i), 'RBF_kernel'};
    [alpha,b] = trainlssvm(parameters);
    test_classes = simlssvm(parameters, {alpha,b}, Xtest);
    miss_rate = sum(test_classes ~= Ytest) / length(Ytest);
    fprintf('Missclassification rate : %.2f\n', miss_rate)
    plotlssvm(parameters, {alpha,b});
    waitforbuttonpress
end