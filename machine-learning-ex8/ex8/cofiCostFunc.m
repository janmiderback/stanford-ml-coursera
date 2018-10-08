function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%% Non-vectorized calc. of J
% for i=1:num_movies
%     for j=1:num_users
%         uparam = Theta(j, :);
%         mfeat = X(i, :);
%         J = J + R(i, j) * (mfeat * uparam - Y(i, j)) ^ 2;
%     end
% end
% J = J / 2;

%% Vectorized calc. of J
J = R .* ((X * Theta' - Y) .^ 2);
J = sum(sum(J)) / 2;

%% Non-vectorized calc. of X_grad
% for i = 1:num_movies
%     for k = 1:num_features       
%         X_grad(i, k) = 0;
%         for j = 1:num_users
%             uparam = Theta(j, :);
%             mfeat = X(i, :);
%             X_grad(i, k) = X_grad(i, k) + R(i, j) * (mfeat * uparam' - Y(i, j)) * Theta(j, k);
%         end
%     end
% end

%% Vectorized calc. of X_grad
X_grad = R .* (X * Theta' - Y);
X_grad = X_grad * Theta;

%% Non-vectorized calc. of Theta_grad
% for j = 1:num_users
%     for k = 1:num_features       
%         Theta_grad(j, k) = 0;
%         for i = 1:num_movies
%             uparam = Theta(j, :);
%             mfeat = X(i, :);
%             Theta_grad(j, k) = Theta_grad(j, k) + R(i, j) * (mfeat * uparam' - Y(i, j)) * X(i, k);
%         end
%     end
% end

%% Vectorized calc. of Theta_grad
Theta_grad = R .* (X * Theta' - Y);
Theta_grad = Theta_grad' * X;

%% Regularized cost function
R1 = sum(sum(Theta .^ 2));
R2 = sum(sum(X .^ 2));
J = J + (lambda / 2) * (R1 + R2);

%% Regularized gradient
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;

%% ============================================================

grad = [X_grad(:); Theta_grad(:)];

end
