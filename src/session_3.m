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
kpca_script_custom
%% 1.2 Spectral clustering
sclustering_script
%% 1.3a Fixed-size LS-SVM
fixedsize_script1
%% 1.3b Fixed-size LS-SVM
fslssvm_script
%% Functions
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