function y = ThermalRSBRabiFlop(nbar,t,Omega,eta)

n_array = 0:2000;
log_P_array = n_array.*log(nbar) - (n_array + 1).*log(nbar+1);
%P_array = (nbar.^n_array)./((nbar+1).^(n_array+1));
P_array = exp(log_P_array);
Omega_array = eta.*Omega.*sqrt(n_array);
t = reshape(t,[numel(t) 1]);
t_array = repmat(t,[1 numel(n_array)]);
P_array = repmat(P_array,[numel(t) 1]);
Omega_array = repmat(Omega_array,[numel(t) 1]);
y_array = P_array.*(sin(Omega_array.*t_array./2)).^2;

y = sum(y_array,2);