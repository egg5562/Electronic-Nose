% OneLDA 함수에서 PCA부분만 빼낸 것

function [data_pca, w_pca, mean_f, std_f, eig_val] = PCA(tr_data, comp_consider, n_pc, out_file)
%tr_data = load('H:\FR\feature_sel/FERET/Experi_data/Feret_tr_fafb_equal_200x2p.dat');
%comp_consider = 1; n_pc = 100; out_file = 'practice';
%tr_file_name = 'Feret_train_100x120_200_FaFb.dat'; N_C = 100; Tr_Each = 2; comp_consider = 1;   %Cov. matrix : N_TrxN_Tr, in case of comp_consider=1
%n_pc = 100;  n_midsel = 100; n_sel = 90; out_file = 'LDA_42-241_140'; data_tr=load(tr_file_name); tr_data=data_tr(1:200,:);

%Read input file
[N_Tr, temp] = size(tr_data);
N_F = temp-1;     %N_Tr = 100*2, %N_F = 100*120
data_tr = tr_data(:,1:N_F);
class_tr = tr_data(:,N_F+1);
clear tr_data;

%Normalize
[mean_f, std_f] = cal_std(data_tr);
for i=1:N_Tr,
   for j=1:N_F,
      if std_f(j)==0
          data_tr(i,j) = 0;
      else
          data_tr(i,j) = (data_tr(i,j)-mean_f(j)) / std_f(j);
      end
      
   end
end
%fid = fopen([out_file,'_mean.dat'], 'w');
%for j=1:N_F,
%   fprintf(fid,'%.4f ', mean_f(j));
%end
%fprintf(fid,'\n');
%for j=1:N_F,
%   fprintf(fid,'%.4f ', std_f(j));
%end
%fprintf(fid,'\n');
%fclose(fid);

%Start the watch for lasting time of the feature extraction
 

%PCA
if comp_consider == 1,
   data_tr_tp = data_tr';
   K = data_tr * data_tr_tp;     %K : (N_Tr) * (N_Tr), Caution that this cov. is correct only if already normailized.
   K = K / N_Tr;
   [q_comp, d] = eig(K);
   ra_K = rank(K);
%    display(ra_K);
   for i=1:N_Tr,
      d_temp(i) = d(i,i);
   end
   %clear K;  clear d;

   q = data_tr_tp * q_comp;
   [eig_val, d_idx] = sort(d_temp);    %sort in ascending order
%    
%    Total_sum = sum(eig_val);
%    eig_val_sum = 0;
%    for i=1:size(eig_val,2)
%    eig_val_sum = eig_val(i) + eig_val_sum;
%    sum_rate = eig_val_sum/Total_sum;
%         if (sum_rate >= 0.97)
%             n_pc = i;       
%             break
%         end
%     end
   
   w_pca = zeros(N_F, n_pc);
   for i=1:n_pc,
      w_pca(:,i) = q(:,d_idx(N_Tr-i+1));   %w_pca : (N_F) * (n_pc), Caution that n_pc should be less than N_Tr in this case.
      norm_weight = norm(w_pca(:,i));
      w_pca(:,i) = w_pca(:,i) / norm_weight;
   end
   clear data_tr_tp;
else
   data_tr_tp = data_tr';
   K = data_tr_tp * data_tr;     %K : (N_F) * (N_F), Caution that this cov. is correct only if already normailized.
   clear data_tr_tp;
   K = K / N_Tr;
   [q, d] = eig(K);
   for i=1:N_F,
      d_temp(i) = d(i,i);
   end
   
   [eig_val, d_idx] = sort(d_temp);
   w_pca = zeros(N_F, n_pc);
   for i=1:n_pc,
      w_pca(:,i) = q(:,d_idx(N_F-i+1));   %w_pca : (N_F) * (n_pc)
   end
end

% [N_OrigF, N_NewF] = size(w_pca);
% fid = fopen([out_file,'_PCAweight.dat'], 'w');
% for i=1:N_OrigF,
%    for j=1:N_NewF,
%       fprintf(fid,'%.4f ', w_pca(i,j));
%    end
%    fprintf(fid,'\n');
% end
% fclose(fid);

data_pca = data_tr * w_pca;
% data_pca = data_pca';
% [eig_val, d_idx] = sort(d_temp,'descend');    %sort in ascending order