gamma_S = 95.3; %Decay rate of P_1/2 state to S_1/2 in MHz
gamma_D = 31; %Decay rate of P_1/2 state to D_3/2 in MHz

%Various components to finally compute the analytical form for the P state population, sigma_PP
AA = Omega_S*gamma_D/Omega_D + Omega_D*gamma_S/Omega_S;
BB = Omega_D/(2*Omega_S) + Omega_S/(2*Omega_D);

Bracket1 = (AA/(2*(Delta_S-Delta_D)))*(-Omega_D*Delta_S/Omega_S + Omega_S*Delta_D/Omega_D)...
    + (BB/(Delta_S-Delta_D))*(-(Omega_D*Delta_S*gamma_S/Omega_S) + (Omega_S*Delta_D*gamma_D/Omega_D))...
    + ((AA*BB)/(4*(Delta_S - Delta_D)^2))*(Omega_D^2 + Omega_S^2) ...
    + (2*Delta_S^2*gamma_S)/(Omega_S^2) + (2*Delta_D^2*gamma_D)/(Omega_D^2);

ImgSPPS = 1i*2*gamma_S/Omega_S;

ImgDPPD = 1i*2*gamma_D/Omega_D;

sigma_PP = 1/(3 + (2/(gamma_S + gamma_D))*Bracket1...
    - 1i*((gamma_S + gamma_D)/(2*Omega_S))*ImgSPPS...
    - 1i*((gamma_S + gamma_D)/(2*Omega_D))*ImgDPPD);