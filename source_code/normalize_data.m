    

function normalized_data = normalize_data(set,m,s)
    
    si= size(set);
    ForRepmat = si(1);
    new_mean = repmat(m,ForRepmat,1);
    new_std = repmat(s,ForRepmat,1);
    normalized_data = (set- new_mean)./new_std;
end

%      si= size(tr_set(:,1:end-1));
%     ForRepmat = si(2);
%     new_mean = repmat(mean_f,ForRepmat,1);
%     new_std = repmat(std_f,ForRepmat,1);
%        
%     normalized_data = (tr_set(:,1:end-1)- new_mean)./new_std;