function [psi] = fn_psi(z, beta, eta)

%% parameters
% beta : inverse temperature
% eta : saturation value

psi = exp(-beta*(z-eta))/(1+exp(-beta*(z-eta)));



end