function [y_next] = fn_irf_review(data_set, w_pca, threshold)

%% Iteratively reweighted Fitting of Eigenfaces
% data_set : test data
% y_t : test_set * w_pca
% w_pca : training w_pca
[n_sample, dim] = size(data_set);

y_t = w_pca' * data_set';

beta = 0.1;     % beta : inverse temperature
eta = 100;      % eta : saturation value


for n = 1 : n_sample
    sub_y_t = y_t(:,n);
    t = 1 ;
    while 1
        y_t_temp = zeros(size(sub_y_t,1), size(sub_y_t,2));
        y_inv = zeros(size(sub_y_t,1), size(sub_y_t,1));
        
        for irf = 1 : dim
            z = ((data_set(n,irf))' - (w_pca(irf,:)*sub_y_t)).^2;
            omega = fn_psi(z, beta, eta);
            y_inv = y_inv + omega * w_pca(irf,:)' * w_pca(irf,:);
            y_t_temp = y_t_temp + omega * w_pca(irf,:)' * (data_set(n,irf))';            
        end
        
        sub_y_next = inv(y_inv) * y_t_temp;
        
        fprintf('num_sample %d(iter %d) : %f\n',n,t,norm(sub_y_next - sub_y_t));
        
        if norm(sub_y_next - sub_y_t) < threshold || t == 1000
            y_next(:,n) = sub_y_next;
            fprintf('num_sample %d done!\n',n);
            break;
        end
        
        sub_y_t = sub_y_next;
        
        t = t + 1 ;
    end
end