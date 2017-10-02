function [randX, randy] = randXy(X, y)

Xy = [X y];
randXy = Xy(randperm(size(Xy, 1)), :);
randX = randXy(:,1:end-1);
randy = randXy(:,end);

end