%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('svm'))
addpath(genpath('figures'))
addpath(genpath('lssvm'))
close all
%% 2.1 Support vector machine for function estimation
N = 20;
X = (1:N)' - .5;
Y = 2*X - 10;
save('data/linear_regression_toy.mat', 'X', 'Y');
uiregress
%% 2.2 A simple example: the sinc function
rng('default') % Fix random seed
X = (-3:0.01:3)';
Y = sinc(X) + 0.1 .* randn(length(X), 1);
Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);
%% 2.2a A simple example: the sinc function (regression)
% demofun
%
i = 1;
for gamma = [10 1E3 1E6]
    for sigma = [1E-2 1 1E2]
        parameters = {Xtrain, Ytrain, 'function estimation', gamma, sigma, 'RBF_kernel'};
        [alpha,b] = trainlssvm(parameters);
        Ysim = simlssvm(parameters, {alpha,b}, Xtest);
        figure
        hold on;
        p = plot(Xtest, Ytest, '.', 'MarkerSize', 10);
        set(p, 'Color', [0, 0.4470, 0.7410]);
        p = plot(Xtest, Ysim, '-', 'LineWidth', 4);
        set(p, 'Color', [0.4660, 0.6740, 0.1880]);
        %legend('Test data', 'Result of regression');
        xlabel('X');
        ylabel('Y');
        mse = mean((Ysim - Ytest) .^ 2);
        fprintf('MSE (%.2f,%.2f) = %.4f\n', gamma, sigma, mse);
        waitforbuttonpress;
        %export_pdf(gcf, sprintf('estimation/regression_%d', i))
        i = i + 1;
        hold off;
    end
end
close all
%% 2.2b A simple example: the sinc function (regression)
parameters = {Xtrain, Ytrain, 'f', [], [], 'RBF_kernel'};
display_runtime = 1;
N = 100; % Number of runs
if display_runtime tic; end %#ok
gsimp = ones(1,N);
ssimp = ones(1,N);
csimp = ones(1,N);
for i = 1:N
    [gsimp(i),ssimp(i),csimp(i)] = tunelssvm(parameters, 'simplex', 'crossvalidatelssvm', {10, 'mse'});
end
fprintf('Average (gamma,sigma2,cost) = (%f,%f,%f)\n', mean(gsimp), mean(ssimp), mean(csimp))
fprintf('Std (gamma,sigma2,cost) = (%f,%f,%f)\n', std(gsimp), std(ssimp), std(csimp))
if display_runtime toc; end %#ok
%
if display_runtime tic; end %#ok
ggrid = ones(1,N);
sgrid = ones(1,N);
cgrid = ones(1,N);
for i = 1:N
    [ggrid(i),sgrid(i),cgrid(i)] = tunelssvm(parameters, 'gridsearch', 'crossvalidatelssvm', {10, 'mse'});
end
fprintf('Average (gamma,sigma2,cost) = (%f,%f,%f)\n', mean(ggrid), mean(sgrid), mean(cgrid))
fprintf('Std (gamma,sigma2,cost) = (%f,%f,%f)\n', std(ggrid), std(sgrid), std(cgrid))
if display_runtime toc; end %#ok
%% 2.2c Application to the Bayesian framework
sigma = 0.4;
gamma = 10;
parameters = {Xtrain, Ytrain, 'f', gamma, sigma};
crit_L1 = bay_lssvm(parameters, 1);
crit_L2 = bay_lssvm(parameters, 2);
crit_L3 = bay_lssvm(parameters, 3);
[~,alpha,b] = bay_optimize(parameters, 1);
[~,gamma] = bay_optimize(parameters, 2);
[~,sigma] = bay_optimize(parameters, 3);
sig2e = bay_errorbar(parameters, 'figure');
fprintf("(alpha, b, gamma, sigma) = (%f,%f,%f,%f)\n", alpha, b, gamma ,sigma);
export_pdf(gcf, 'estimation/bayesian_inference')
%% 2.3 Automatic Relevance Determination
X = 6 .* rand(100, 3) - 3;
Y = sinc(X(:,1)) + 0.1 .* randn(100,1); % Only the first column is not random
sigma = 0.1; % Choose parameters wisely to prevent it from modelling noise
gamma = 10;
%[selected, ranking] = bay_lssvmARD({X, Y, 'f', gamma, sigma}) %#ok
subsets = (dec2bin(1:2^3-1) - '0')';
costs = [];
for set = subsets
    parameters = {X(:, set > 0), Y, 'f', gamma, sigma, 'RBF_kernel'};
    costs = [costs crossvalidate(parameters, 10, 'mse', 'mean')]; %#ok
end
figure
hold on
colors = flipud(lines(7));
for c = 1:3
    [Xsort,sort_indices] = sort(X(:,c));
    plot(Xsort, Y(sort_indices), '-', 'Color', colors(c,:), 'LineWidth', 2*(c==1)+1)
end
legend('X1', 'X2', 'X3')
hold off
export_pdf(gcf, 'estimation/ard')
%% 2.4 Robust regression
rng('default')
X = (-6:0.2:6)';
Y = sinc(X) + 0.1 .* rand(size(X));
out = [15 17 19];
Y(out) = 0.7 + 0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5 + 0.2*rand(size(out));
%% 2.4a Robust regression (non-robust)
model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'crossvalidatelssvm';
model = tunelssvm(model, 'simplex', costFun, {10, 'mae';});
model = plotlssvm(model);
export_pdf(gcf, 'estimation/robustregression_non')
figure
p1 = plot(1:length(model.alpha), model.alpha ./ model.gam, '.', 'MarkerSize', 15);
xlabel('index')
ylabel('\alpha_k/\gamma')
set(p1, 'Color', [0, 0.4470, 0.7410]);
export_pdf(gcf, 'estimation/robustregression_non_alpha')
%% 2.4b Robust regression (robust)
model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFuns = ["whuber" "whampel" "wlogistic" "wmyriad"];
times = ones(length(wFuns), 1);
i = 1;
for wFun = wFuns
    tic
    model = tunelssvm(model, 'simplex', costFun, {10, 'mae';}, wFun);
    model = robustlssvm(model);
    times(i) = toc;
    i = i + 1;
    plotlssvm(model);
    Ysim = simlssvm(model, Xtest);
    mse = mean((Ysim - Ytest) .^ 2);
    fprintf('Mse error for %s = %.4f', wFun, mse);
    export_pdf(gcf, sprintf('estimation/robustregression_%s', wFun))
    figure
    p1 = plot(1:length(model.alpha), model.alpha ./ (model.gam)', 'o');
    xlabel('index')
    ylabel('\alpha_k/\gamma')
    set(p1, 'Color', [0, 0.4470, 0.7410]);
    export_pdf(gcf, sprintf('estimation/robustregression_%s_alpha', wFun))
end