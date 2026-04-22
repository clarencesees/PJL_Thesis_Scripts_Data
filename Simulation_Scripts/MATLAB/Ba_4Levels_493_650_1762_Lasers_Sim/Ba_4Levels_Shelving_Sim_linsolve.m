%This script calculates a steady-state solution of the S_1/2, P_1/2, D_3/2 and D_5/2 state populations of a simplified 4-level barium energy level structure.
%S denotes the S_1/2 state.
%P denotes the P_1/2 state.
%D1 denotes the D_3/2 state.
%D2 denotes the D_5/2 state.

%Define the Rabi frequencies of the 493 (Omega_SP), 650 (Omega_D1P) and 1762 nm (Omega_SD2) lasers.
Omega_SP = 2*pi*10;
Omega_D1P = 2*pi*10;
Omega_SD2 = 2*pi*0.1*0;

%Define the frequency detunings of 493 (Delta_SP), 650 (Delta_D1P) and 1762 nm (Delta_SD2) lasers.
Delta_SP = -2*pi*10;
Delta_D1P = -2*pi*40;
%Delta_SD2 = -2*pi*0;

gamma_S = 95.3; %Decay rate of P_1/2 state to S_1/2 in MHz
gamma_D = 31; %Decay rate of P_1/2 state to D_3/2 in MHz

%Define basis states as vectors.
s1 = zeros(1,4);
s1(1) = 1;

d1 = zeros(1,4);
d1(2) = 1;

d2 = zeros(1,4);
d2(3) = 1;

p1 = zeros(1,4);
p1(4) = 1;

cs1 = kron(s1,ones(1,4));
cs1_conj = kron(ones(1,4),s1);

cd1 = kron(d1,ones(1,4));
cd1_conj = kron(ones(1,4),d1);

cd2 = kron(d2,ones(1,4));
cd2_conj = kron(ones(1,4),d2);

cp1 = kron(p1,ones(1,4));
cp1_conj = kron(ones(1,4),p1);

ind_array = eye(numel(s1));

c_array = nan(numel(s1),numel(cs1));
c_conj_array = nan(numel(s1),numel(cs1));

for c_array_count=1:numel(s1)
    c_array(c_array_count,:) = kron(ind_array(c_array_count,:),ones(1,numel(s1)));
    c_conj_array(c_array_count,:) = kron(ones(1,numel(s1)),ind_array(c_array_count,:));
end

%Define the time derivatives of each state probability amplitude.
cs1_prime = -1i.*(Delta_SP).*cs1...
    -1i.*(Omega_SP/2).*cp1...
    -1i.*(Omega_SD2/2).*cd2;
cs1_prime_conj = conj(-1i.*(Delta_SP).*cs1_conj...
    -1i.*(Omega_SP/2).*cp1_conj...
    -1i.*(Omega_SD2/2).*cd2_conj);

cd1_prime = -1i.*(Delta_D1P).*cd1...
    -1i.*(Omega_D1P/2).*cp1;
cd1_prime_conj = conj(-1i.*(Delta_D1P).*cd1_conj...
    -1i.*(Omega_D1P/2).*cp1_conj);

cd2_prime = -1i.*(Delta_SP - Delta_SD2).*cd2...
    -1i.*(Omega_SD2/2).*cs1;
cd2_prime_conj = conj(-1i.*(Delta_SP - Delta_SD2).*cd2_conj...
    -1i.*(Omega_SD2/2).*cs1_conj);

cp1_prime = -1i.*(Omega_SP/2).*cs1...
    -1i.*(Omega_D1P/2).*cd1;
cp1_prime_conj = conj(-1i.*(Omega_SP/2).*cs1_conj...
    -1i.*(Omega_D1P/2).*cd1_conj);

%Aggregate the state amplitude time derivative vectors into a 2D array.
c_prime_array = [...
    cs1_prime;...
    cd1_prime;...
    cd2_prime;...
    cp1_prime];
    
c_prime_conj_array = [...
    cs1_prime_conj;...
    cd1_prime_conj;...
    cd2_prime_conj;...
    cp1_prime_conj];

Evol_Matrix = nan(numel(cs1),numel(cs1)); %Initialize the matrix for 
%state evolution.

%Construct the time-evolution matrix for the density matrix.
for EM_count1 = 1:numel(s1)
    for EM_count2 = 1:numel(s1)
        Pcount = 0;
        if EM_count1 == 4
            Pcount = Pcount+1;
        end
        if EM_count2 == 4
            Pcount = Pcount+1;
        end
        Evol_Matrix(c_array(EM_count1,:).*c_conj_array(EM_count2,:) == 1,:) =...
            c_array(EM_count1,:).*c_prime_conj_array(EM_count2,:) + c_prime_array(EM_count1,:).*c_conj_array(EM_count2,:)...
            - (1/2).*Pcount.*(gamma_S + gamma_D).*c_array(EM_count1,:).*c_conj_array(EM_count2,:);
    end
end     

Evol_Matrix(cs1.*cs1_conj == 1,:) = cs1.*cs1_prime_conj + cs1_prime.*cs1_conj...
    + gamma_S.*cp1.*cp1_conj;

Evol_Matrix(cd1.*cd1_conj == 1,:) = cd1.*cd1_prime_conj + cd1_prime.*cd1_conj...
    + gamma_D.*cp1.*cp1_conj;

Evol_Matrix(cd2.*cd2_conj == 1,:) = cd2.*cd2_prime_conj + cd2_prime.*cd2_conj;

%Add another line to the Evol_Matrix to be used for constructing the restriction that the trace of the density matrix is 1.
Evol_Matrix(end+1,:) = zeros(1,size(Evol_Matrix,2));
for EM_count=1:numel(s1)
    Evol_Matrix(end,c_array(EM_count,:).*c_conj_array(EM_count,:) == 1) = 1;
end

%Solve for the steady-state density matrix.
sigma_end = linsolve(Evol_Matrix,[zeros(size(Evol_Matrix,2),1); 1]);