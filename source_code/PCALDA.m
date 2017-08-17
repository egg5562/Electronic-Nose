%	1DLDA
%	by Choi Sang-il
%	12. Sep. 2005
%   eigen valueï¿½éµµ return ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ (06.06.01)
%   06.11.10  OneLDAï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ (line 26=> class_inform ï¿½ï¿½ ï¿½×¶ï¿½ï¿½×¶ï¿½ ï¿½Ô·ï¿½ï¿½ï¿½ï¿½Ö¾ï¿½ï¿?ï¿½ï¿½, line 56~74, line 119~150 ï¿½ß°ï¿½,   line 152~165   inactive)
%   ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?Modi_OneLDA.mï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½, returnï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ï¿½Ï°ï¿½ flag_pca ï¿½ï¿½ï¿½Î¿ï¿½ ï¿½Ô·ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½Ù²ï¿½ ï¿½ï¿½ LDA.mï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½
%   08.12.05  LDA.m ï¿½ï¿½ ï¿½Ì¸ï¿½ï¿½ï¿½ PCALDA.mï¿½ï¿½ï¿½ï¿½ ï¿½Ù²ï¿½ï¿½ï¿½
%   comp_consider ï¿½ß°ï¿½ --> param(5)
%   08.12.16  parameterï¿½ï¿½ï¿½ï¿½ param vectorï¿½ï¿½ ï¿½ï¿½ï¿½Ðµï¿½ï¿?
%   n_pc : param(1),  n_midsel : param(2),   n_sel : param(3),   flag_pca = param(4),   comp_consider : param(5)
%   Cbeig_val ï¿½ï¿½ return ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ ï¿½ß°ï¿½

% Modi_OneLDA.mï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ Cw ï¿½ï¿½ê°ªï¿½ï¿?ï¿½à°£ ï¿½ï¿½ï¿½ï¿½ ï¿½Ù¸ï¿½
function [w_final, Cbeig_val, mean_f, std_f] = PCALDA(tr_data, param, out_file)
%clear all; clc;tr_data = load('Feret_tr_fafb_equal_200x2p.dat'); out_file = 'prac';param = [150,150,150,1,1];


% parameters
n_pc = param(1); % PCA¿¡¼­ »ç¿ëÇÏ´Â Æ¯Â¡º¤ÅÍ °³¼ö
n_midsel = param(2); % pca¿¡¼­ »ç¿ëÇÏ´Â Æ¯Â¡º¤ÅÍ °³¼ö
n_sel = param(3); % LDA¿¡¼­ »ç¿ëÇÏ´Â Æ¯Â¡º¤ÅÍ °³¼ö
flag_pca = param(4);
comp_consider = param(5);

% data sorting according to class label(ascending order)
class = tr_data(:,end);
[N_C,class_label,N_class_sample]=class_information(class);     % class label: classï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ ex) 1,2,3,4.. N_sample_class: ï¿½ï¿½ ï¿½ï¿½ï¿½Â¿ï¿½ï¿½ï¿½ï¿½ï¿½ classï¿½ï¿½ sampleï¿½ï¿½
[sorted_data,class_index] = class_separation(tr_data,class,class_label,N_C);
[N_Tr,N_F] = size(sorted_data(:,1:end-1)); clear tr_data;
data_tr = sorted_data(:,1:end-1);
class_tr = sorted_data(:,end);
clear sorted_data;
class_inform = class_label';

%Normalize
[mean_f, std_f] = cal_std(data_tr);
data_tr = data_normalize(data_tr, mean_f, std_f);

display('data loading over!');
if flag_pca == 1,
    if comp_consider == 1,
        % PCA
        data_tr_tp = data_tr';
        K = data_tr * data_tr_tp;     %K : (N_Tr) x (N_Tr), Caution that this cov. is correct only if already normailized.
        K = K / N_Tr;
        [q_comp, d] = eig(K);
        for i=1:N_Tr,
            d_temp(i) = d(i,i);
        end
        clear K;  clear d;

        q = data_tr_tp * q_comp;
        [eig_val, d_idx] = sort(d_temp);    %sort in ascending order
        w_pca = zeros(N_F, n_pc);
        for i=1:n_pc,
            w_pca(:,i) = q(:,d_idx(N_Tr-i+1));   %w_pca : (N_F) x (n_pc), Caution that n_pc should be less than N_Tr in this case.
            norm_weight = norm(w_pca(:,i));
            w_pca(:,i) = w_pca(:,i) / norm_weight;
        end
        clear data_tr_tp;
        data_pca = data_tr * w_pca;        % data_tr : N_tr x N_F ,  w_pca : N_F x n_pc      , data_pca : N_tr x n_pc
        %display('PCA process over!');
    else
        data_tr_tp = data_tr';
        K = data_tr_tp * data_tr;     %K : (N_F) x (N_F), Caution that this cov. is correct only if already normailized.
        clear data_tr_tp;
        K = K / N_Tr;
        [q, d] = eig(K);
        for i=1:N_F,
            d_temp(i) = d(i,i);
        end
        [eig_val, d_idx] = sort(d_temp);
        w_pca = zeros(N_F, n_pc);
        for i=1:n_pc,
            w_pca(:,i) = q(:,d_idx(N_F-i+1));   %w_pca : (N_F) x (n_pc)
        end
        data_pca = data_tr * w_pca;        
    end
else,
    data_pca = data_tr;
    n_pc = N_F;
    n_midsel = n_pc;
end

% class ï¿½ï¿½ï¿½ï¿½ data ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
class_1 = data_pca(1:N_class_sample(1),:);       % class_1 : size = N_clsss_samle(1) x N_F
start_idx = N_class_sample(1) + 1;
for i = 2:N_C,
    end_idx = start_idx + N_class_sample(i) - 1;
    eval(['class_',num2str(i),' = data_pca(start_idx:end_idx,:);']);
    start_idx = start_idx + N_class_sample(i);
end                            % data_pca = [class_1; class_2;,,,];

% mean for each class
sum_class = zeros(N_C, n_pc);
mean_class = zeros(N_C, n_pc);
for i = 1:N_C,                          % N_C : #. of class
    for j = 1:N_class_sample(i),
        eval(['sum_class(i,:) = sum_class(i,:) + class_',num2str(i),'(j,:);']);
    end
    mean_class(i,:) = sum_class(i,:)/N_class_sample(i);
end

%Within-cov. matrix, Cw calculating
%************* Modi_OneLDA ï¿½ï¿½ Cw ï¿½ï¿½ï¿?ï¿½ï¿½ï¿?*******************
%Cw_2 = zeros(n_pc, n_pc); 
%Cw_1 = zeros(n_pc, n_pc);
%for i = 1:N_class_sample(1),
%    data_rest_1 = data_pca(i,:) - mean_class(1,:);
%    Cw_1 = Cw_1 + (data_rest_1'*data_rest_1);
%end
%for i = 2:N_C,
%    temp8=sum(N_class_sample(1:(i-1)))+1;
%    temp9=sum(N_class_sample(1:i));
%    for j=temp8:temp9,
%        data_rest = data_pca(j,:) - mean_class(i,:);
%        Cw_2 = Cw_2 + (data_rest_2'*data_rest_2);            % Cw : n_pc x n_pc
%    end
%end
%another_Cw = Cw_2+Cw_1;
%another_Cw = another_Cw / N_Tr;
%*************************************************************

Cw = zeros(n_pc, n_pc);
for i = 1:N_C
    for j = 1:N_class_sample(i),
        eval(['data_rest = class_',num2str(i),'(j,:) - mean_class(i,:);']);
        Cw = Cw + data_rest' * data_rest;
    end
end
Cw = Cw / N_Tr;
display(['Cw rank = ',num2str(rank(Cw))]);
if rank(Cw) == n_pc,
    display('Cw has full rank');
    n_midsel = rank(Cw);
end
display('Sw calculation over!');



%Between-cov. matrix, Cb calculating
% Calculating Cb
[mean_tot, std_tot] = cal_std(data_pca);
Cb = zeros(n_pc, n_pc);
for i = 1:N_C,
    rest_data = mean_class(i,:) - mean_tot;
    Cb = Cb + N_class_sample(i) * rest_data' * rest_data;
end
Cb = Cb / N_C;
display(['Cb rank = ',num2str(rank(Cb)), ' = n_midsel' ]);

%Eigenvalue decomposition of Cw
[q_1tot, d_1tot] = eig(Cw);
for i=1:n_pc,
    d1_value(i) = d_1tot(i,i);
end
[Cweig_val, d_idx] = sort(d1_value);   %sort in ascending order

for i=1:n_midsel,                            % n_midsel <= n_pc
    q_1(:,i) = q_1tot(:,d_idx(n_pc-i+1));     % n_pc x n_midsel
    d_1(i) = Cweig_val(n_pc-i+1);
end

d_invroot = zeros(n_midsel,n_midsel);        % n_midsel x n_midsel
for i=1:n_midsel,
    d_invroot(i,i) = sqrt(1/d_1(i));            % whiteningï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½ ï¿½Û¾ï¿½
end
w_1 = q_1 * d_invroot;       % w_1 : n_pc x n_midsel

Cw_1 = w_1' * Cw * w_1;      % Cw_1 : Identity matrix
Cb_1 = w_1' * Cb * w_1;      % whitening
% Cb_1 : n_midsel x n_midsel    Cb : n_pc x n_pc

%Eigenvalue decomposition of Cb_1
[q_2, d_2] = eig(Cb_1);
for i=1:n_midsel,
    d2_value(i) = d_2(i,i);
end
[Cbeig_val, d_idx] = sort(d2_value);   %sort in ascending order

for i=1:n_sel,
    w_2(:,i) = q_2(:,d_idx(n_midsel-i+1));     % n_midsel x n_sel
    %d_3(i) = Cbeig_val(n_midsel-i+1);          %% 06.06.01
end

display(['ra_Cb = ', num2str(rank(Cb)), '& ra_Cb_1 = ' ,num2str(rank(Cb_1))]);
display('1DLDA end');

fid = fopen([out_file,'_ldaeig.dat'], 'w');
fprintf(fid,'<PCA eigenvalues>\n');
if flag_pca == 1,
    for i=1:n_pc,
        fprintf(fid,'%.4f ', eig_val(end-i+1));
    end
    fprintf(fid,'\n\n');
    eig_tot = sum(eig_val);
    for i=2:2:n_pc,
        eig_extracted = sum(eig_val(end-i+1:end));
        eig_rate_vec(i) = (eig_extracted/eig_tot) * 100;
        fprintf(fid,'eig_rate(%d features) : %.2f\n', i, eig_rate_vec(i));
    end
    fprintf(fid,'\n\n');
end
fprintf(fid,'<Cw eigenvalues>\n');
for i=1:n_midsel,
    fprintf(fid,'%.4f ', Cweig_val(end-i+1));
end
fprintf(fid,'\n\n');
eig_tot = sum(Cweig_val);
for i=10:10:n_midsel,
    eig_extracted = sum(Cweig_val(end-i+1:end));
    eig_rate_vec(i) = (eig_extracted/eig_tot) * 100;
    fprintf(fid,'eig_rate(%d features) : %.2f\n', i, eig_rate_vec(i));
end
fprintf(fid,'\n\n');
fprintf(fid,'<Cb_1 eigenvalues>\n');
for i=1:n_sel,
    fprintf(fid,'%.4f ', Cbeig_val(end-i+1));
end
fprintf(fid,'\n\n');
eig_tot = sum(Cbeig_val);
for i=10:10:n_sel,
    eig_extracted = sum(Cbeig_val(end-i+1:end));
    eig_rate_vec(i) = (eig_extracted/eig_tot) * 100;
    fprintf(fid,'eig_rate(%d features) : %.2f\n', i, eig_rate_vec(i));
end
fprintf(fid,'\n\n');
fclose(fid);

w_lda = w_1 * w_2;       % (n_pc x n_sel) = (n_pc x n_midsel) * (n_midsel x n_sel)
if flag_pca == 1,
    w_final = w_pca * w_lda;    % (N_tr x n_sel) = (N_tr x n_pc) * (n_pc x n_sel)
else       
    w_final = w_lda;
end

%[N_OrigF, N_NewF] = size(w_final);
%fid = fopen([out_file,'_weight.dat'], 'w');
%for i=1:N_OrigF,
%    for j=1:N_NewF,
%        fprintf(fid,'%.4f ', w_final(i,j));
%    end
%    fprintf(fid,'\n');
%end
%fclose(fid);

%Tr. data projection
%tr_prj = data_pca * w_lda;
%clear data_pca;

%Renormalization
%[mean_final, std_final] = cal_std(tr_prj);
%fid = fopen([out_file,'_meanfinal.dat'], 'w');
%for j=1:N_NewF,
%    fprintf(fid,'%.4f ', mean_final(j));
%end
%fprintf(fid,'\n');
%for j=1:N_NewF,
%    fprintf(fid,'%.4f ', std_final(j));
%end
%fprintf(fid,'\n');
%fclose(fid);