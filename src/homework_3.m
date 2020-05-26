%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
addpath(genpath('svm'))
close all
%% 2.1 Kernel Principal Component Analysis
digitsdn_custom
%% 2.2 Fixed-size LS-SVM (shuttle)
%fslssvm_script
data = load('shuttle.dat','-ascii');
%data = data(1:700,:);
X = data(:,1:end-1);
Y = data(:,end);
un = unique(Y)' %#ok
%colormap(parula)
% for i = 1:9
%     figure
%     hold on
%     for j = un
%         histogram(X(Y==j,i))
%     end
%     hold off
% end
for j = un
    fprintf('Number of instances in class %i = %i\n', i, sum(Y == j))
end
%% 2.3 Fixed-size LS-SVM
%fslssvm_script
data = load('california.dat','-ascii');
X = data(:,1:end-1);
Y = data(:,end);
