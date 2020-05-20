load iris


%
% train LS-SVM classifier with linear kernel 
%
type='c'; 
gam = 1; 
disp('Linear kernel'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)

disp('Press any key to continue...'), pause, 




%
% Train the LS-SVM classifier using polynomial kernel
%
type='c'; 
gam = 1; 
t = 1; 
degree = 5;
disp('Polynomial kernel of degree 5'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
disp('Press any key to continue...'), pause,        
    

%
% use RBF kernel
%

% tune the sig2 while fix gam
%
disp('RBF kernel')
gam = 1; sig2list=[0.01, 0.1, 1, 5, 10, 25];

errlist=[];

for sig2=sig2list,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause,         
end


%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(sig2list), errlist, '*-'), 
xlabel('log(sig2)'), ylabel('number of misclass'),



