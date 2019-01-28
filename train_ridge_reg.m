function [weight, min_val, max_val] = train_ridge_reg(X, Y, d, lambda)

%% removing bad points
all_train_data = [X; Y];
bad_idx = find( sum(all_train_data ) < -5000 );
X(:,bad_idx) = [];
Y(:,bad_idx) = [];

%% scaling data
min_val = min(X,[],2);
max_val = max(X,[],2);

X = (X - repmat(min_val,1,size(X,2)) ) ./ repmat( max_val-min_val+1, 1, size(X,2) );
% K = (X'*X+1).^d;
% K_te = (X'*X_te+1).^d;
% N = size(Y,1);
% pred = Y / (K+lambda*eye(N)) * K_te;

D = size(X,1);

if d == 1
    new_X = X; 
end

if d >= 2
    new_X = X;
    
    for i = 1:D
        for j = i:D
            new_X = [new_X; X(i,:).*X(j,:)];
        end
    end
end


new_X = [new_X; ones(1,size(new_X,2))];


weight = inv(new_X*new_X'+lambda*eye(size(new_X,1)))*new_X*Y';