function [error_train, error_val] = ...
      learningCurveRandom(X, y, Xval, yval, lambda)

  % Number of training examples
  m = size(X, 1);
  r = size(Xval, 1)
  % function return values
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);

  for i = 1 : m
    for j = 1 : 50
      selt = randperm(m, i);
      selv = randperm(r, i);
      Xt = X(selt,:);
      yt = y(selt,:);
      Xv = Xval(selv,:);
      yv = yval(selv,:);
      theta = trainLinearReg(Xt, yt, lambda);
      error_train(i) += linearRegCostFunction(Xt, yt, theta, 0);
      error_val(i) += linearRegCostFunction(Xv, yv, theta, 0);
    end
    error_train(i) /= 50;
    error_val(i) /= 50;
  end
end
