%% Initialisation
clc
clear
addpath(genpath('data'))
addpath(genpath('lssvm'))
addpath(genpath('figures'))
close all
%% 2.1 Ripley dataset
load('ripley')
visualise(Xtrain, Ytrain)
export_pdf(gcf, 'classification/ripley')
%% 2.2 Breast Cancer dataset
load('breast')
visualise(trainset, labels_train)
export_pdf(gcf, 'classification/wisconsin')
%% 2.3 Diabetes dataset
load('diabetes')
visualise(trainset, labels_train)
export_pdf(gcf, 'classification/diabetes')
%% Functions
function visualise(X, y)
    figure
    hold on
    plot(X(y>0,1), X(y>0,2), 'bo','MarkerFaceColor','b');
    plot(X(y<0,1), X(y<0,2), 'ro','MarkerFaceColor','r');
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