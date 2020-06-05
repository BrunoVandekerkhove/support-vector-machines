function perf=rmse(e)
%
% calculate the root mean squared error of the given errors
% 
%  'perf = rmse(E);'
%
% see also:
%    mse, mae, linf, trimmedmse
%

% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab


perf = sqrt(sum(sum(e.^2)) / numel(e));