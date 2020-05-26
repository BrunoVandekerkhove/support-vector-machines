%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
addpath(genpath('svm'))
close all
%% 2.2a Logmap dataset (copy-paste from document)
load('logmap')
% Turn into regression (future in function of past values)
order = 10;
X = windowize(Z, 1:(order + 1));
Y = X(:, end);
X = X(:, 1:order);
%
gam = 10;
sig2 = 10;
%[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});
% Define starting point (last point of training set) and predict
Xs = Z(end-order+1:end, 1); % past values for starting point
nb = 50;
prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);
% Visualise
figure;
hold on;
n=plot(Ztest, 'k', 'LineWidth', 2);
n.set('Color', [0, 0.4470, 0.7410]);
p=plot(prediction, 'r', 'LineWidth', 2);
p.set('Color', [0.6350, 0.0780, 0.1840]);
hold off;
%export_pdf(gcf, 'logmap/init', 1440, 120)
%% 2.2b Logmap dataset (tuning)
load('logmap')
res = [];
errors = [];
for kernel = ["RBF_kernel"]
    for order = 1:10
        X = windowize(Z, 1:(order + 1));
        Y = X(:, end);
        X = X(:, 1:order);
        %
        tune_params = {X, Y, 'f', 0.1, [], kernel};
        [gam,params,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10,'mse'});
        gam = 1.0;
        params = 1.0;
        % define starting point (last point of training set) and predict
        res = [res ; gam, params, cost]; %#ok
        Xs = Z(end-order+1:end, 1); % past values for starting point
        nb = length(Ztest); 
        truth = Ztest;
        Xs = Z(1:order,1);
        nb = size(X,1);
        truth = Y;
        prediction = predict({X, Y, 'f', gam, params}, Xs, nb);
        error = mse(prediction - truth);
        errors = [errors ; error]; %#ok
        fprintf('MSE on test set = %.5f', error)
        % visualise
%         figure;
%         hold on;
%         plot(Ztest, 'k');
%         plot(prediction, 'r');
%         hold off;
        %export_pdf(gcf, sprintf('logmap/pred_%s_%d', kernel, order), 1440, 170)
        %close all
    end
end
%% 2.3a Santa Fe dataset (experiments)
load('santafe')
res = [];
errors = [];
for kernel = ["RBF_kernel"]
    for order = 1:50
        X = windowize(Z, 1:(order + 1));
        Y = X(:, end);
        X = X(:, 1:order);
        %
        tune_params = {X, Y, 'f', 0.1, [], kernel};
        [gam,params,cost] = tunelssvm(tune_params, 'simplex', 'leaveoneoutlssvm', {'mse'});
        % define starting point (last point of training set) and predict
        res = [res ; gam, params, cost]; %#ok
        Xs = Z(end-order+1:end, 1); % past values for starting point
        nb = length(Ztest);
        truth = Ztest;
        prediction = predict({X, Y, 'f', gam, params}, Xs, nb);
        error = mse(prediction - truth);
        errors = [errors ; error]; %#ok
        fprintf('MSE on test set = %.5f', error)
        % visualise
%         figure;
%         hold on;
%         plot(Ztest, 'k');
%         plot(prediction, 'r');
%         hold off;
        %export_pdf(gcf, sprintf('logmap/pred_%s_%d', kernel, order), 1440, 170)
        %close all
    end
end
%% Misc (logmap)
clear
clc
load('logmap')
maes = [];
mses = [];
for order = 21
    % Turn into regression (future in function of past values)
    %order = 2;
    X = windowize(Z, 1:(order + 1));
    Y = X(:, end);
    X = X(:, 1:order);
    %
    %tune_params = {X, Y, 'f', [], [], 'RBF_kernel'};
    %[gamma,sigma,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10,'mse'});
    %[alpha,b] = trainlssvm({X, Y, 'f', gamma, sigma, 'RBF_kernel'});
    sigma = 1;
    gamma = 10;
    [~,alpha,b] = bay_optimize({X, Y, 'f', gamma, sigma}, 1);
    [~,gamma] = bay_optimize({X, Y, 'f', gamma, sigma}, 2);
    [~,sigma] = bay_optimize({X, Y, 'f', gamma, sigma}, 3);
    %
    to_predict = Ztest;
    Xs = Z(end-order+1:end,1);
    nb = length(to_predict);
    %
%     to_predict = Y;
%     Xs = X(1,:);
%     nb = size(Y,1);
    %
    prediction = predict({X, Y, 'f', gamma, sigma, 'RBF_kernel'}, Xs, nb);
    mae = mean(abs(to_predict - prediction)) %#ok
    maes = [maes ; mae]; %#ok
    mse = mean((to_predict - prediction).^2) %#ok
    mses = [mses ; mse]; %#ok
    % Visualise
    figure;
    hold on;
    n=plot(to_predict, 'k', 'LineWidth', 2);
    n.set('Color', [0, 0.4470, 0.7410]);
    p=plot(prediction, 'r', 'LineWidth', 2);
    p.set('Color', [0.6350, 0.0780, 0.1840]);
    hold off;
    set(gcf, 'Position', [0 300 1440 120])
    %export_pdf(gcf, 'logmap/init', 1440, 120)
    % waitforbuttonpress
end
%% Misc (santafe)
clear
clc
load('santafe')
maes = [];
mses = [];
for order = 42
    % Turn into regression (future in function of past values)
    %order = 2;
    X = windowize(Z, 1:(order + 1));
    Y = X(:, end);
    X = X(:, 1:order);
    %
    tune_params = {X, Y, 'f', [], [], 'RBF_kernel'};
    [gamma,sigma,cost] = tunelssvm(tune_params, 'simplex', 'crossvalidatelssvm', {10,'mse'});
    [alpha,b] = trainlssvm({X, Y, 'f', gamma, sigma, 'RBF_kernel'});
    %sigma = 1;
    %gamma = 1;
    %[~,alpha,b] = bay_optimize({X, Y, 'f', gamma, sigma}, 1);
    %[~,gamma] = bay_optimize({X, Y, 'f', gamma, sigma}, 2);
    %[~,sigma] = bay_optimize({X, Y, 'f', gamma, sigma}, 3);
    %
    to_predict = Ztest;
    Xs = Z(end-order+1:end,1);
    nb = length(to_predict);
    %
%     to_predict = Y;
%     Xs = X(1,:);
%     nb = size(Y,1);
    %
    prediction = predict({X, Y, 'f', gamma, sigma, 'RBF_kernel'}, Xs, nb);
    mae = mean(abs(to_predict - prediction)) %#ok
    maes = [maes ; mae]; %#ok
    mse = mean((to_predict - prediction).^2) %#ok
    mses = [mses ; mse]; %#ok
    %plot(maes, 'r-')
    % Visualise
    figure;
    hold on;
    n=plot(to_predict, 'k', 'LineWidth', 2);
    n.set('Color', [0, 0.4470, 0.7410]);
    p=plot(prediction, 'r', 'LineWidth', 2);
    p.set('Color', [0.6350, 0.0780, 0.1840]);
    hold off;
%     set(gcf, 'Position', [0 300 1440 120])
%     %export_pdf(gcf, 'logmap/init', 1440, 120)
%     waitforbuttonpress
end
hold off