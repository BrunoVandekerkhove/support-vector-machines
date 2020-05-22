function perf = rsplitvalidate(model, perc, measure)

% Check input arguments.
if length(model) ~= 6
    error('rsplitvalidate: model should be cell array with 6 entries.');
end

if size(model{1}, 1) ~= size(model{2}, 1)
    error('rsplitvalidate: sizes training data and labels do not match.');
end

if ~strcmp(model{3}, 'c')
    error('rsplitvalidate: function only implemented for classification case, not for function estimation.');
end

if ~(strcmp(model{6}, 'lin_kernel') || strcmp(model{6}, 'poly_kernel') || strcmp(model{6}, 'RBF_kernel'))
    error('rsplitvalidate: invalid kernel input.');
end 

if strcmp(model{6}, 'poly_kernel')
    if ~(size(model{5}, 1) == 2 && size(model{5}, 2) == 1)
        error('rsplitvalidate: polynomial kernel should be characterized by two kernel parameters: t and degree d.');
    end
end 

if strcmp(model{6}, 'RBF_kernel')
    if ~(size(model{5}, 1) == 1 && size(model{5}, 2) == 1)
        error('rsplitvalidate: RBF kernel should be characterized by one kernel parameter: sig2.');
    end
end
    
 
% Collect arguments.
Xall = model{1};
Yall = model{2};
mfl  = model{3};
gam  = model{4};
kpar = model{5};
ker  = model{6};
 
% Random permutation.
idx = randperm(size(Xall, 1));
 
% Percentage recalculation.
ptrain = floor(perc*length(Xall));

% Assign training and validation data.
Xtrain = Xall(idx(1:ptrain),:);
Ytrain = Yall(idx(1:ptrain));
Xval   = Xall(idx(ptrain + 1:end),:);
Yval   = Yall(idx(ptrain + 1:end),:);

% Train the model.
[alpha, b] = trainlssvm({Xtrain, Ytrain, mfl, gam, kpar, ker}); 
 
% Simulate the model.
estY = simlssvm({Xtrain, Ytrain, mfl, gam, kpar, ker}, {alpha, b}, Xval);

% Output.
if strcmp(measure, 'misclass')
    % Misclassification error.
    perf = sum(estY ~= Yval)/length(Yval);
else
    error('rsplitvalidate: Only misclass measure currently implemented.');
end