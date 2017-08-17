%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Mean and Std calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mean_vec, std_vec] = cal_std(data);

[N_P, N_F] = size(data);

for j=1:N_F,
   sum(j) = 0;
   for i=1:N_P,
      sum(j) = sum(j) + data(i,j);
   end
   mean_vec(j) = sum(j) / N_P;
end

for j=1:N_F,
   sum(j) = 0;
   for i=1:N_P,
      sum(j) = sum(j) + (data(i,j)-mean_vec(j))^2;
   end
   std_vec(j) = sqrt(sum(j)/(N_P-1));
end
