function pet(X, Y, Xt, Yt, ker, alpha, bias) % to be used with SVM toolbox, prints error percentage on test set
    error = svcerror(X, Y, Xt, Yt, ker, alpha, bias);
    fprintf('Error for kernel %s = %.5f\n', ker, error/length(Xt));
end