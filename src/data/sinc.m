function Y = sinc(X)
    Y = sin(pi.*X+12345*eps) ./ (pi*X+12345*eps);
end