Planck_h = 6.62607015*(10^(-34)); %Planck constant in SI units.
uB = 9.274009994*10^(-24); %Bohr magneton in SI units.
B = 1*470*10^(-6); %Magnetic field strength in Gauss.
Delta_zs = 10^(-6)*2*uB*B/Planck_h; %S level Zeeman splitting in MHz.
Delta_zd = 10^(-6)*(4/5)*uB*B/Planck_h; %D level Zeeman splitting in MHz.
Delta_zp = 10^(-6)*(2/3)*uB*B/Planck_h; %P level Zeeman splitting in MHz
pol_minus_s = sqrt((cos(theta)^2)/2); pol_0_s = sqrt(sin(theta)^2); pol_plus_s = sqrt((cos(theta)^2)/2); %S-P laser polarization components.
pol_minus_d = sqrt((cos(theta)^2)/2); pol_0_d = sqrt(sin(theta)^2); pol_plus_d = sqrt((cos(theta)^2)/2); %D-P laser polarizatoin components.

%pol_minus_s = 0; pol_0_s = 1; pol_plus_s = 0;
%pol_minus_d = sqrt(1/2); pol_0_d = 0; pol_plus_d = sqrt(1/2);

gamma_S = 95.3; %Decay rate of P_1/2 state to S_1/2 in MHz
gamma_D = 31; %Decay rate of P_1/2 state to D_3/2 in MHz
%Omega_S = Omega_S_multiplier*899.4180;
%Omega_S2 = 0.3*899.4180;
%Omega_D0 = 0.3*1.4169e+03;
%Omega_D = Omega_D_multiplier*1.4169e+03;
%Omega_D2 = 0.3*1.4169e+03;
%Omega_D3 = 0.3*1.4169e+03;
%Delta_S1 = -2*pi*20;
%Delta_S2 = -2*pi*40;
%Delta_D0 = -2*pi*200;
%Delta_D1 = -2*pi*200;
%Delta_D2 = -2*pi*200;
%Delta_D3 = -2*pi*200;

%Define energy states as vector arrays.
cs0 = kron([1 0 0 0 0 0 0 0],ones(1,8));
cs0_conj = kron(ones(1,8),[1 0 0 0 0 0 0 0]);
cs1 = kron([0 1 0 0 0 0 0 0],ones(1,8));
cs1_conj = kron(ones(1,8),[0 1 0 0 0 0 0 0]);
cd0 = kron([0 0 1 0 0 0 0 0],ones(1,8));
cd0_conj = kron(ones(1,8),[0 0 1 0 0 0 0 0]);
cd1 = kron([0 0 0 1 0 0 0 0],ones(1,8));
cd1_conj = kron(ones(1,8),[0 0 0 1 0 0 0 0]);
cd2 = kron([0 0 0 0 1 0 0 0],ones(1,8));
cd2_conj = kron(ones(1,8),[0 0 0 0 1 0 0 0]);
cd3 = kron([0 0 0 0 0 1 0 0],ones(1,8));
cd3_conj = kron(ones(1,8),[0 0 0 0 0 1 0 0]);
cp0 = kron([0 0 0 0 0 0 1 0],ones(1,8));
cp0_conj = kron(ones(1,8),[0 0 0 0 0 0 1 0]);
cp1 = kron([0 0 0 0 0 0 0 1],ones(1,8));
cp1_conj = kron(ones(1,8),[0 0 0 0 0 0 0 1]);

%Compute first derivatives of energy state amplitudes.
cs0_prime = 1i.*(Omega_S/2).*(sqrt(1/3).*pol_0_s.*cp0 + sqrt(2/3).*pol_plus_s.*cp1);
cs0_prime_conj = conj(1i.*(Omega_S/2).*(sqrt(1/3).*pol_0_s.*cp0_conj + sqrt(2/3).*pol_plus_s.*cp1_conj));

cs1_prime = -1i.*Delta_zs.*cs1...
    - 1i.*(Omega_S/2).*(sqrt(2/3).*pol_minus_s.*cp0 + sqrt(1/3).*pol_0_s.*cp1);
cs1_prime_conj = conj(-1i.*Delta_zs.*cs1_conj...
    - 1i.*(Omega_S/2).*(sqrt(2/3).*pol_minus_s.*cp0_conj + sqrt(1/3).*pol_0_s.*cp1_conj));

cd0_prime = 1i.*(Delta_SP01 - Delta_DP01).*cd0...
    -1i.*(Omega_D/2).*sqrt(1/2).*pol_plus_d.*cp0;
cd0_prime_conj = conj(1i.*(Delta_SP01 - Delta_DP01).*cd0_conj...
    -1i.*(Omega_D/2).*sqrt(1/2).*pol_plus_d.*cp0_conj);

cd1_prime = 1i.*(Delta_SP01 - Delta_DP01 - Delta_zd).*cd1...
    - 1i.*(Omega_D/2).*(sqrt(1/6).*pol_plus_d.*cp1 - sqrt(1/3).*pol_0_d.*cp0);
cd1_prime_conj = conj(1i.*(Delta_SP01 - Delta_DP01 - Delta_zd).*cd1_conj...
    - 1i.*(Omega_D/2).*(sqrt(1/6).*pol_plus_d.*cp1_conj - sqrt(1/3).*pol_0_d.*cp0_conj));

cd2_prime = 1i.*(Delta_SP01 - Delta_DP01 - 2.*Delta_zd).*cd2...
    - 1i.*(Omega_D/2).*(sqrt(1/6).*pol_minus_d.*cp0 - sqrt(1/3).*pol_0_d.*cp1);
cd2_prime_conj = conj(1i.*(Delta_SP01 - Delta_DP01 - 2.*Delta_zd).*cd2_conj...
    - 1i.*(Omega_D/2).*(sqrt(1/6).*pol_minus_d.*cp0_conj - sqrt(1/3).*pol_0_d.*cp1_conj));

cd3_prime = 1i.*(Delta_SP01 - Delta_DP01 - 3.*Delta_zd).*cd3...
    - 1i.*(Omega_D/2).*sqrt(1/2).*pol_minus_d.*cp1;
cd3_prime_conj = conj(1i.*(Delta_SP01 - Delta_DP01 - 3.*Delta_zd).*cd3_conj...
    - 1i.*(Omega_D/2).*sqrt(1/2).*pol_minus_d.*cp1_conj);

cp0_prime = 1i.*(Delta_SP01 + Delta_zp).*cp0...
    - 1i.*(Omega_S/2).*(sqrt(2/3).*pol_minus_s.*cs1 - sqrt(1/3).*pol_0_s.*cs0)...
    - 1i.*(Omega_D/2).*(sqrt(1/2).*pol_plus_d.*cd0 - sqrt(1/3).*pol_0_d.*cd1 + sqrt(1/6).*pol_minus_d.*cd2);
cp0_prime_conj = conj(1i.*(Delta_SP01 + Delta_zp).*cp0_conj...
    - 1i.*(Omega_S/2).*(sqrt(2/3).*pol_minus_s.*cs1_conj - sqrt(1/3).*pol_0_s.*cs0_conj)...
    - 1i.*(Omega_D/2).*(sqrt(1/2).*pol_plus_d.*cd0_conj - sqrt(1/3).*pol_0_d.*cd1_conj + sqrt(1/6).*pol_minus_d.*cd2_conj));

cp1_prime = 1i.*Delta_SP01.*cp1...
    - 1i.*(Omega_S/2).*(sqrt(1/3).*pol_0_s.*cs1 - sqrt(2/3).*pol_plus_s.*cs0)...
    - 1i.*(Omega_D/2).*(sqrt(1/6).*pol_plus_d.*cd1 - sqrt(1/3).*pol_0_d.*cd2 + sqrt(1/2).*pol_minus_d.*cd3);
cp1_prime_conj = conj(1i.*Delta_SP01.*cp1_conj...
    - 1i.*(Omega_S/2).*(sqrt(1/3).*pol_0_s.*cs1_conj - sqrt(2/3).*pol_plus_s.*cs0_conj)...
    - 1i.*(Omega_D/2).*(sqrt(1/6).*pol_plus_d.*cd1_conj - sqrt(1/3).*pol_0_d.*cd2_conj + sqrt(1/2).*pol_minus_d.*cd3_conj));

%Construct the matrix for time-evolution.
Evol_Matrix = nan(numel(cs0),numel(cs0));

Evol_Matrix(cs0.*cs0_conj == 1,:) = cs0.*cs0_prime_conj + cs0_prime.*cs0_conj...
    + (1/3).*gamma_S.*cp0.*cp0_conj + (2/3).*gamma_S.*cp1.*cp1_conj;

Evol_Matrix(cs1.*cs1_conj == 1,:) = cs1.*cs1_prime_conj + cs1_prime.*cs1_conj...
    + (2/3).*gamma_S.*cp0.*cp0_conj + (1/3).*gamma_S.*cp1.*cp1_conj;

Evol_Matrix(cd0.*cd0_conj == 1,:) = cd0.*cd0_prime_conj + cd0_prime.*cd0_conj...
    + (1/2).*gamma_D.*cp0.*cp0_conj;

Evol_Matrix(cd1.*cd1_conj == 1,:) = cd1.*cd1_prime_conj + cd1_prime.*cd1_conj...
    + (1/3).*gamma_D.*cp0.*cp0_conj + (1/6).*gamma_D.*cp1.*cp1_conj;

Evol_Matrix(cd2.*cd2_conj == 1,:) = cd2.*cd2_prime_conj + cd2_prime.*cd2_conj...
    + (1/6).*gamma_D.*cp0.*cp0_conj + (1/3).*gamma_D.*cp1.*cp1_conj;

Evol_Matrix(cd3.*cd3_conj == 1,:) = cd3.*cd3_prime_conj + cd3_prime.*cd3_conj...
    + (1/2).*gamma_D.*cp1.*cp1_conj;

Evol_Matrix(cp0.*cp0_conj == 1,:) = cp0.*cp0_prime_conj + cp0_prime.*cp0_conj...
    - (gamma_S + gamma_D).*cp0.*cp0_conj;

Evol_Matrix(cp1.*cp1_conj == 1,:) = cp1.*cp1_prime_conj + cp1_prime.*cp1_conj...
    - (gamma_S + gamma_D).*cp1.*cp1_conj;

Evol_Matrix(cs0.*cs1_conj == 1,:) = cs0.*cs1_prime_conj + cs0_prime.*cs1_conj;
Evol_Matrix(cs1.*cs0_conj == 1,:) = cs1.*cs0_prime_conj + cs1_prime.*cs0_conj;

Evol_Matrix(cs0.*cd0_conj == 1,:) = cs0.*cd0_prime_conj + cs0_prime.*cd0_conj;
Evol_Matrix(cd0.*cs0_conj == 1,:) = cd0.*cs0_prime_conj + cd0_prime.*cs0_conj;

Evol_Matrix(cs0.*cd1_conj == 1,:) = cs0.*cd1_prime_conj + cs0_prime.*cd1_conj;
Evol_Matrix(cd1.*cs0_conj == 1,:) = cd1.*cs0_prime_conj + cd1_prime.*cs0_conj;

Evol_Matrix(cs0.*cd2_conj == 1,:) = cs0.*cd2_prime_conj + cs0_prime.*cd2_conj;
Evol_Matrix(cd2.*cs0_conj == 1,:) = cd2.*cs0_prime_conj + cd2_prime.*cs0_conj;

Evol_Matrix(cs0.*cd3_conj == 1,:) = cs0.*cd3_prime_conj + cs0_prime.*cd3_conj;
Evol_Matrix(cd3.*cs0_conj == 1,:) = cd3.*cs0_prime_conj + cd3_prime.*cs0_conj;

Evol_Matrix(cs0.*cp0_conj == 1,:) = cs0.*cp0_prime_conj + cs0_prime.*cp0_conj...
    - ((gamma_S + gamma_D)./2).*cs0.*cp0_conj;
Evol_Matrix(cp0.*cs0_conj == 1,:) = cp0.*cs0_prime_conj + cp0_prime.*cs0_conj...
    - ((gamma_S + gamma_D)./2).*cp0.*cs0_conj;

Evol_Matrix(cs0.*cp1_conj == 1,:) = cs0.*cp1_prime_conj + cs0_prime.*cp1_conj...
    - ((gamma_S + gamma_D)./2).*cs0.*cp1_conj;
Evol_Matrix(cp1.*cs0_conj == 1,:) = cp1.*cs0_prime_conj + cp1_prime.*cs0_conj...
    - ((gamma_S + gamma_D)./2).*cp1.*cs0_conj;

Evol_Matrix(cs1.*cd0_conj == 1,:) = cs1.*cd0_prime_conj + cs1_prime.*cd0_conj;
Evol_Matrix(cd0.*cs1_conj == 1,:) = cd0.*cs1_prime_conj + cd0_prime.*cs1_conj;

Evol_Matrix(cs1.*cd1_conj == 1,:) = cs1.*cd1_prime_conj + cs1_prime.*cd1_conj;
Evol_Matrix(cd1.*cs1_conj == 1,:) = cd1.*cs1_prime_conj + cd1_prime.*cs1_conj;

Evol_Matrix(cs1.*cd2_conj == 1,:) = cs1.*cd2_prime_conj + cs1_prime.*cd2_conj;
Evol_Matrix(cd2.*cs1_conj == 1,:) = cd2.*cs1_prime_conj + cd2_prime.*cs1_conj;

Evol_Matrix(cs1.*cd3_conj == 1,:) = cs1.*cd3_prime_conj + cs1_prime.*cd3_conj;
Evol_Matrix(cd3.*cs1_conj == 1,:) = cd3.*cs1_prime_conj + cd3_prime.*cs1_conj;

Evol_Matrix(cs1.*cp0_conj == 1,:) = cs1.*cp0_prime_conj + cs1_prime.*cp0_conj...
    - ((gamma_S + gamma_D)./2).*cs1.*cp0_conj;
Evol_Matrix(cp0.*cs1_conj == 1,:) = cp0.*cs1_prime_conj + cp0_prime.*cs1_conj...
    - ((gamma_S + gamma_D)./2).*cp0.*cs1_conj;

Evol_Matrix(cs1.*cp1_conj == 1,:) = cs1.*cp1_prime_conj + cs1_prime.*cp1_conj...
    - ((gamma_S + gamma_D)./2).*cs1.*cp1_conj;
Evol_Matrix(cp1.*cs1_conj == 1,:) = cp1.*cs1_prime_conj + cp1_prime.*cs1_conj...
    - ((gamma_S + gamma_D)./2).*cp1.*cs1_conj;

Evol_Matrix(cd0.*cd1_conj == 1,:) = cd0.*cd1_prime_conj + cd0_prime.*cd1_conj;
Evol_Matrix(cd1.*cd0_conj == 1,:) = cd1.*cd0_prime_conj + cd1_prime.*cd0_conj;

Evol_Matrix(cd0.*cd2_conj == 1,:) = cd0.*cd2_prime_conj + cd0_prime.*cd2_conj;
Evol_Matrix(cd2.*cd0_conj == 1,:) = cd2.*cd0_prime_conj + cd2_prime.*cd0_conj;

Evol_Matrix(cd0.*cd3_conj == 1,:) = cd0.*cd3_prime_conj + cd0_prime.*cd3_conj;
Evol_Matrix(cd3.*cd0_conj == 1,:) = cd3.*cd0_prime_conj + cd3_prime.*cd0_conj;

Evol_Matrix(cd0.*cp0_conj == 1,:) = cd0.*cp0_prime_conj + cd0_prime.*cp0_conj...
    - ((gamma_S + gamma_D)./2).*cd0.*cp0_conj;
Evol_Matrix(cp0.*cd0_conj == 1,:) = cp0.*cd0_prime_conj + cp0_prime.*cd0_conj...
    - ((gamma_S + gamma_D)./2).*cp0.*cd0_conj;

Evol_Matrix(cd0.*cp1_conj == 1,:) = cd0.*cp1_prime_conj + cd0_prime.*cp1_conj...
    - ((gamma_S + gamma_D)./2).*cd0.*cp1_conj;
Evol_Matrix(cp1.*cd0_conj == 1,:) = cp1.*cd0_prime_conj + cp1_prime.*cd0_conj...
    - ((gamma_S + gamma_D)./2).*cp1.*cd0_conj;

Evol_Matrix(cd1.*cd2_conj == 1,:) = cd1.*cd2_prime_conj + cd1_prime.*cd2_conj;
Evol_Matrix(cd2.*cd1_conj == 1,:) = cd2.*cd1_prime_conj + cd2_prime.*cd1_conj;

Evol_Matrix(cd1.*cd3_conj == 1,:) = cd1.*cd3_prime_conj + cd1_prime.*cd3_conj;
Evol_Matrix(cd3.*cd1_conj == 1,:) = cd3.*cd1_prime_conj + cd3_prime.*cd1_conj;

Evol_Matrix(cd1.*cp0_conj == 1,:) = cd1.*cp0_prime_conj + cd1_prime.*cp0_conj...
    - ((gamma_S + gamma_D)./2).*cd1.*cp0_conj;
Evol_Matrix(cp0.*cd1_conj == 1,:) = cp0.*cd1_prime_conj + cp0_prime.*cd1_conj...
    - ((gamma_S + gamma_D)./2).*cp0.*cd1_conj;

Evol_Matrix(cd1.*cp1_conj == 1,:) = cd1.*cp1_prime_conj + cd1_prime.*cp1_conj...
    - ((gamma_S + gamma_D)./2).*cd1.*cp1_conj;
Evol_Matrix(cp1.*cd1_conj == 1,:) = cp1.*cd1_prime_conj + cp1_prime.*cd1_conj...
    - ((gamma_S + gamma_D)./2).*cp1.*cd1_conj;

Evol_Matrix(cd2.*cd3_conj == 1,:) = cd2.*cd3_prime_conj + cd2_prime.*cd3_conj;
Evol_Matrix(cd3.*cd2_conj == 1,:) = cd3.*cd2_prime_conj + cd3_prime.*cd2_conj;

Evol_Matrix(cd2.*cp0_conj == 1,:) = cd2.*cp0_prime_conj + cd2_prime.*cp0_conj...
    - ((gamma_S + gamma_D)./2).*cd2.*cp0_conj;
Evol_Matrix(cp0.*cd2_conj == 1,:) = cp0.*cd2_prime_conj + cp0_prime.*cd2_conj...
    - ((gamma_S + gamma_D)./2).*cp0.*cd2_conj;

Evol_Matrix(cd2.*cp1_conj == 1,:) = cd2.*cp1_prime_conj + cd2_prime.*cp1_conj...
    - ((gamma_S + gamma_D)./2).*cd2.*cp1_conj;
Evol_Matrix(cp1.*cd2_conj == 1,:) = cp1.*cd2_prime_conj + cp1_prime.*cd2_conj...
    - ((gamma_S + gamma_D)./2).*cp1.*cd2_conj;

Evol_Matrix(cd3.*cp0_conj == 1,:) = cd3.*cp0_prime_conj + cd3_prime.*cp0_conj...
    - ((gamma_S + gamma_D)./2).*cd3.*cp0_conj;
Evol_Matrix(cp0.*cd3_conj == 1,:) = cp0.*cd3_prime_conj + cp0_prime.*cd3_conj...
    - ((gamma_S + gamma_D)./2).*cp0.*cd3_conj;

Evol_Matrix(cd3.*cp1_conj == 1,:) = cd3.*cp1_prime_conj + cd3_prime.*cp1_conj...
    - ((gamma_S + gamma_D)./2).*cd3.*cp1_conj;
Evol_Matrix(cp1.*cd3_conj == 1,:) = cp1.*cd3_prime_conj + cp1_prime.*cd3_conj...
    - ((gamma_S + gamma_D)./2).*cp1.*cd3_conj;

Evol_Matrix(cp0.*cp1_conj == 1,:) = cp0.*cp1_prime_conj + cp0_prime.*cp1_conj...
    - (gamma_S + gamma_D).*cp0.*cp1_conj;
Evol_Matrix(cp1.*cp0_conj == 1,:) = cp1.*cp0_prime_conj + cp1_prime.*cp0_conj...
    - (gamma_S + gamma_D).*cp1.*cp0_conj;

%Add an additional constraint for trace of density matrix being 1.
Evol_Matrix(end+1,:) = zeros(1,64);
Evol_Matrix(end,cs0.*cs0_conj==1) = 1;
Evol_Matrix(end,cs1.*cs1_conj==1) = 1;
Evol_Matrix(end,cp0.*cp0_conj==1) = 1;
Evol_Matrix(end,cp1.*cp1_conj==1) = 1;
Evol_Matrix(end,cd0.*cd0_conj==1) = 1;
Evol_Matrix(end,cd1.*cd1_conj==1) = 1;
Evol_Matrix(end,cd2.*cd2_conj==1) = 1;
Evol_Matrix(end,cd3.*cd3_conj==1) = 1;

%Find steady-state solution.
sigma_end = linsolve(Evol_Matrix,[zeros(64,1); 1]);