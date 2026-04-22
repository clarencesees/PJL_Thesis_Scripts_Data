%Omega = 2*pi*100;
%Delta_SP = -2*pi*1000;

%Polarization
sigma_pi_percentage = 5;
sigma_minus_percentage = 0;
sigma_plus_percentage = 100-sigma_pi_percentage-sigma_minus_percentage;

sigma_pi = sqrt(sigma_pi_percentage/100);
sigma_minus = sqrt(sigma_minus_percentage/100);
sigma_plus = sqrt(sigma_plus_percentage/100);

%Rabi frequencies in MHz
Omega_S0P0 = sigma_pi*Omega;
Omega_S0P1 = sigma_plus*Omega;
Omega_S1P0 = sigma_minus*Omega;
Omega_S1P1 = sigma_pi*Omega;

gamma_S = 95.3; %Decay rate of P_1/2 state to S_1/2 in MHz

%Define quantum states as vector arrays.
s0 = zeros(1,4);
s0(1) = 1;

s1 = zeros(1,4);
s1(2) = 1;

p0 = zeros(1,4);
p0(3) = 1;

p1 = zeros(1,4);
p1(4) = 1;

cs0 = kron(s0,ones(1,4));
cs0_conj = kron(ones(1,4),s0);

cs1 = kron(s1,ones(1,4));
cs1_conj = kron(ones(1,4),s1);

cp0 = kron(p0,ones(1,4));
cp0_conj = kron(ones(1,4),p0);

cp1 = kron(p1,ones(1,4));
cp1_conj = kron(ones(1,4),p1);

ind_array = eye(numel(s0));

c_array = nan(numel(s0),numel(cs0));
c_conj_array = nan(numel(s0),numel(cs0));

for c_array_count=1:numel(s0)
    c_array(c_array_count,:) = kron(ind_array(c_array_count,:),ones(1,numel(s0)));
    c_conj_array(c_array_count,:) = kron(ones(1,numel(s0)),ind_array(c_array_count,:));
end

%Compute first derivatives of energy state amplitudes.
cs0_prime = -1i.*(Delta_SP).*cs0...
    -1i.*(Omega_S0P0/2).*cp0...
    -1i.*(Omega_S0P1/2).*cp1;
cs0_prime_conj = conj(-1i.*(Delta_SP).*cs0_conj...
    -1i.*(Omega_S0P0/2).*cp0_conj...
    -1i.*(Omega_S0P1/2).*cp1_conj);

cs1_prime = -1i.*(Delta_SP).*cs1...
    -1i.*(Omega_S1P0/2).*cp0...
    -1i.*(Omega_S1P1/2).*cp1;
cs1_prime_conj = conj(-1i.*(Delta_SP).*cs1_conj...
    -1i.*(Omega_S1P0/2).*cp0_conj...
    -1i.*(Omega_S1P1/2).*cp1_conj);

cp0_prime = -1i.*(Omega_S0P0/2).*cs0...
    -1i.*(Omega_S1P0/2).*cs1;
cp0_prime_conj = conj(-1i.*(Omega_S0P0/2).*cs0_conj...
    -1i.*(Omega_S1P0/2).*cs1_conj);

cp1_prime = -1i.*(Omega_S0P1/2).*cs0...
    -1i.*(Omega_S1P1/2).*cs1;
cp1_prime_conj = conj(-1i.*(Omega_S0P1/2).*cs0_conj...
    -1i.*(Omega_S1P1/2).*cs1_conj);

c_prime_array = [...
    cs0_prime;...
    cs1_prime;...
    cp0_prime;...
    cp1_prime];
    
c_prime_conj_array = [...
    cs0_prime_conj;...
    cs1_prime_conj;...
    cp0_prime_conj;...
    cp1_prime_conj];

Evol_Matrix = nan(numel(cs0),numel(cs0)); %Initialize the matrix for 
%state evolution.

%Construct the matrix for time-evolution.
for EM_count1 = 1:numel(s0)
    for EM_count2 = 1:numel(s0)
        Pcount = 0;
        if EM_count1 == 3 || 4
            Pcount = Pcount+1;
        end
        if EM_count2 == 3 || 4
            Pcount = Pcount+1;
        end
        Evol_Matrix(c_array(EM_count1,:).*c_conj_array(EM_count2,:) == 1,:) =...
            c_array(EM_count1,:).*c_prime_conj_array(EM_count2,:) + c_prime_array(EM_count1,:).*c_conj_array(EM_count2,:)...
            - (1/2).*Pcount.*gamma_S.*c_array(EM_count1,:).*c_conj_array(EM_count2,:);
    end
end     

Evol_Matrix(cs0.*cs0_conj == 1,:) = cs0.*cs0_prime_conj + cs0_prime.*cs0_conj...
    + 0.5.*gamma_S.*cp0.*cp0_conj + 0.5.*gamma_S.*cp1.*cp1_conj;

Evol_Matrix(cs1.*cs1_conj == 1,:) = cs1.*cs1_prime_conj + cs1_prime.*cs1_conj...
    + 0.5.*gamma_S.*cp0.*cp0_conj + 0.5.*gamma_S.*cp1.*cp1_conj;

%Add an additional constraint for trace of density matrix being 1.
Evol_Matrix(end+1,:) = zeros(1,size(Evol_Matrix,2));
for EM_count=1:numel(s0)
    Evol_Matrix(end,c_array(EM_count,:).*c_conj_array(EM_count,:) == 1) = 1;
end

%Find steady-state solution.
sigma_end = linsolve(Evol_Matrix,[zeros(size(Evol_Matrix,2),1); 1]);