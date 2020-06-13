function x = ANOVA_kernel(a, b, params)
% Radial Basis Function (RBF) kernel function for implicit higher dimension mapping
%
%  X = RBF_kernel(a,b,sig2)
%
% 'sig2' contains the SQUARED variance of the RBF function:
%    X = exp(-||a-b||.^2/sig2)
%  
% 'a' can only contain one datapoint in a row, 'b' can contain N
% datapoints of the same dimension as 'a'. If the row-vector 'sig2'
% contains i=1 to 'dimension' values, each dimension i has a separate 'sig2(i)'.
%
% see also:
%    poly_kernel, lin_kernel, MLP_kernel, trainlssvm, simlssvm

% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab

sigma = params(1);
degree = params(2);

x = zeros(size(b,1),1);
for i = 1:size(b,1)
    x(i,1) = sum(exp(-sigma * (a-b(i,:)).^2).^degree);        
end