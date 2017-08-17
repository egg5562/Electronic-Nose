clc; clear all; close all
addpath('I:\Master_degree\e-nose\source code');
warning off
%% data load
DATA = load ('ori_e_nose_data.dat');
test_label = load ('test_label.csv');  % cross validation을 위한 라벨 설정

%% data specification
num_data = size(DATA,1);
dim = size(DATA,2) - 1;
fold = 8;

%% parameters
missing_ratio = [0.1, 0.2, 0.3, 0.4];    %% 수동으로 바꿀 파라메터
% missing_ratio = [0.4];    %% 수동으로 바꿀 파라메터
time_step = 2000;    %% 2000개 간격으로 센서가 바뀜
div_step = 100;  %% 10Hz로 2초 주기로 손실 데이터 부여
sensor_num = dim / time_step; % 20개

%% 8 fold cross validation

tic
loss_recog = [];
loss_idx = [];

for mr = 1 : length(missing_ratio)
    
    total_recog = [];
    iter_recog = [];
    missing_idx = [];
    rng(600); % 100, 600
    for cv = 1 : fold
        
        shuf_idx  = randperm(num_data);  % 데이터 개수만큼 순열 설정
        for number = 1 : size(shuf_idx,2)
            shuf_data(number,:) = DATA(shuf_idx(number),:);  % 설정한 순열로 데이터 순서 섞음
        end
        sort_data = sortrows(shuf_data,32001); % 섞은 데이터를 다시 정렬
        
        
        
        stage_recog = [];
        total_idx = [];
        error_mean = [];
        
        for set = 1 : fold
            %% test set 추출
            t_label = zeros(num_data,1);
            t_label(test_label(set,:)) = 1;
            t_label = logical(t_label);
            test_set = sort_data(t_label,:);
            tr_label = logical(1 - t_label);
            
            %% tr set 추출
            tr_set = sort_data(tr_label,:);
            
            %% 특징 추출
            [mean_f, std_f] = cal_std(tr_set(:,1:end-1));
            
	    %% Normalization
            norm_tr_set  = normalize_data(tr_set(:,1:end-1),mean_f,std_f);
            norm_tr_set(:,end+1) = tr_set(:,end);
            
	    %% L1 PCA 실행 
            Ns = size(tr_set,1)-1;
            [w_pca, n_it, elap_time, tr_prj] = L1PCA(norm_tr_set(:,1:end-1), Ns); % 함수 내부에서 normalizaiton 안함
            n_pc = Ns;
            
            m_mean_f = repmat(mean_f,size(test_set,1),1);  % mean_f 확장
            m_std_f = repmat(std_f,size(test_set,1),1);  % mean_f 확장

            %% Add missing data
            test_miss_set = zeros(size(test_set,1), size(test_set,2)-1);
            for k = 1 : size(test_set,1)
                b = zeros(sensor_num,time_step);
                for l = 1 : sensor_num
                    a = test_set(k,1:end-1);
                    b(l,:) = a((time_step*(l-1))+1:time_step*l);
                end
                b = b';
                miss_label = [1:div_step];
                miss_label = repmat(miss_label,time_step/size(miss_label,2),1);
                miss_label = reshape(miss_label,1,time_step);
                b(:,sensor_num+1) = miss_label';
                
                %% 데이터 누락
                
                t_label = zeros(time_step,1);
                miss_rand = randperm(div_step,round(div_step*missing_ratio(1,mr)));
                
                for o = 1 : length(miss_rand)
                    miss_label = find(b(:,end)==miss_rand(1,o));
                    t_label(miss_label) = 1;
                end
                t_label = logical(t_label);
                b(t_label,1:end-1) = 0;
                b(:,sensor_num+1) = [];
                
                %% 데이터 재구성
                test_miss_set(k,:) = reshape(b,1,dim);
            end
            test_miss_set(:,end+1) = test_set(:,end);
            %% Test set Normalization
            norm_test_set  = normalize_data(test_miss_set(:,1:end-1),mean_f,std_f);
            norm_test_set(:,end+1) = test_set(:,end);
            
            
            %% Iteratively reweighted Fitting of Eigenfaces
            
            %         y_t = w_pca' * norm_test_set(:,1:end-1)';
            threshold = 10^(-1);
            [y_next] = fn_irf_review(norm_test_set(:,1:end-1), w_pca, threshold);
            
            
            
            %% RMS
            min_error = [];
            error = zeros(1,n_pc);
            for n = 1 : n_pc
                w = w_pca(:,1:n);
                recon_x = (m_std_f'.*(w*y_next(1:n,:))) + m_mean_f';      % 복원식
                total_rms = sqrt(sum((test_set(:,1:end-1) - recon_x').^2, 2))./dim;  %L2-norm rms
                mean_rms = mean(total_rms);
                error(n) = mean_rms;
            end
            min_error = [min_error; error];
            error_mean(set,:) = min_error;
            
            
            [temp, idx] = min(min_error);
            total_idx = [total_idx idx];
            
            %% Minimum Error
            
            fprintf('The index of minimum error : %d\n', idx);
            w = w_pca(:,1:idx);
            recon_x = (m_std_f'.*(w*y_next(1:idx,:))) + m_mean_f';        %
            recon_x = recon_x';
            
            recon_x(:,end+1) = test_set(:,end);
            
            %% feature Extraction for classification
            new_tr_set = tr_set;
            new_test_set = recon_x;
            out_file = 'prac';
            param = [105 7 7 1 1]; %   n_pc : sample num,  n_midsel : within class num , n_sel : class - 1,   flag_pca = use pca 1, comp_consider : normalize 1
            
            [w_final, Cbeig_val, mean_f, std_f] = PCALDA(new_tr_set,  param, out_file);
            %         [w_final, eig_val_com, mean_f, std_f] = Null_LDA(new_tr_set, out_file);
            
            %% Normalization
            
            new_tr_set(:,1:end-1)  = normalize_data(new_tr_set(:,1:end-1),mean_f,std_f);
            new_test_set(:,1:end-1)  = normalize_data(new_test_set(:,1:end-1),mean_f,std_f);
            
            %% 비교를 위한 사영
            proj_test = w_final' * new_test_set(:,1:end-1)';
            proj_tr = w_final' * new_tr_set(:,1:end-1)';
            proj_test = proj_test';
            proj_tr = proj_tr';
            
            %% Calculate percentage
            
            count = 0;
            for a = 1 : size(proj_test,1)
                temp = [];
                for b = 1 : size(proj_tr,1)
                    temp_val = norm(proj_test(a,:) - proj_tr(b,:));
                    temp = [temp temp_val];
                end
                
                [smallest index] = min(temp);
                
                %최단거리를 갖는 인덱스의 클래스의 값을 갖도록 합니다.
                tr_class = new_tr_set(index,end);
                
                %정확도 계산을 위해 위에서 준 클래스 값과 실제 클래스 값을 비교하여
                %총 몇개가 맞는지 세어 줍니다.
                
                if  new_test_set(a,end) == tr_class
                    count = count +1;
                end
            end
            
            recog = (count / size(new_test_set,1)) *100;
            
            stage_recog = [stage_recog recog];
            disp(['<<<<<< iter(' num2str(set) ') fold complete >>>>>>'])
        end
        set2_recog = sum(stage_recog)/length(stage_recog);
        disp([' ' num2str(set2_recog) '%'])
        
        total_error(cv,:) = mean(error_mean);
        
        missing_idx = [missing_idx; total_idx;];
        iter_recog = [iter_recog set2_recog];
        disp(['--------iter(' num2str(cv) ') complete--------'])
    end
    total_recog = sum(iter_recog)/length(iter_recog);
    disp([' ' num2str(total_recog) '%'])
    
    loss_recog = [loss_recog total_recog];
    loss_idx = [loss_idx; missing_idx;];
end
toc







%% eig_val plot
%     figure(1)
%     eig_cum = cumsum(eig_val)/sum(eig_val)*100;
%     plot(eig_cum,'--.'); grid on;
%     title('Portion of Eigen Values');
%     xlabel('Order of Eigen Values');
%     ylabel('Portion');
% %
%      figure(2)
%     plot([1:length(total_error)],total_error(1,:),'-r.',[1:length(total_error)],total_error(2,:),'-go',[1:length(total_error)],total_error(3,:),'-b^', [1:length(total_error)],total_error(4,:),'-mx' );%,[1:139],error_mean(5,:),'-ch');
%     title('RMS error');
%     xlabel('Number of Eigenvalues');
%     ylabel('Error');
%     legend('10% loss','20% loss','30% loss','40% loss',1);
%     [minimum, idxs] = min(total_error');




figure(3)
subplot(3,1,1);
plot(test_set(1,1:end-1),'r'); grid on;
xlabel('(a)', 'FontSize', 20);
%     legend('Raw Data Sample', 1);
subplot(3,1,2);
plot(test_miss_set(1,1:end-1),'b'); grid on;
xlabel('(b)', 'FontSize', 20);
%     legend('Missed Data Sample', 1);
subplot(3,1,3);
plot(recon_x(1,1:end-1),'b'); hold on;
plot(test_set(1,1:end-1),'r'); grid on;
xlabel('(c)', 'FontSize', 20);
%     legend('Reconstructed Data Sample', 1);