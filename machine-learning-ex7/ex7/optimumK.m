function K = optimumK(S, X)

[~, n] = size(X);
S_n = sum(sum(S));

tol = 0.05;
for i = 1:n
    S_k = sum(sum(S(1:i,1:i)));
    error = 1 - (S_k / S_n);
    if error <= tol
        K = i;
        break
    end
end