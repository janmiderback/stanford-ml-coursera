function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% For each example
for i = 1:size(X, 1)
    x = X(i, :);
    % Start with first centroid.
    c = centroids(1, :);
    min_idx = 1;
    min_dist = norm(x - c);
    % For each centroid, find closest
    for j = 2:K
        c = centroids(j, :);
        dist = norm(x - c);
        if dist < min_dist
            min_idx = j;
            min_dist = dist;
        end
    end
    idx(i) = min_idx;
end

% =============================================================

end
