%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('svm'))
addpath(genpath('figures'))
addpath(genpath('lssvm'))
close all
%% 1.1 Kernel principal component analysis
kpca_script
%% 1.2 Spectral clustering
sclustering_script