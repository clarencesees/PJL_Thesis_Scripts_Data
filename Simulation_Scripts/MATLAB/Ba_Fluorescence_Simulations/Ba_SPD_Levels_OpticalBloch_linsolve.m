gamma_S = 95.3; %Decay rate of P_1/2 state to S_1/2 in MHz
gamma_D = 31; %Decay rate of P_1/2 state to D_3/2 in MHz

%Define basis states as vectors.
cs0 = kron([1 0 0],ones(1,3));
cs0_conj = kron(ones(1,3),[1 0 0]);
cd0 = kron([0 1 0],ones(1,3));
cd0_conj = kron(ones(1,3),[0 1 0]);
cp0 = kron([0 0 1],ones(1,3));
cp0_conj = kron(ones(1,3),[0 0 1]);

%Define the time derivatives of each state probability amplitude.
cs0_prime = -1i.*Delta_S.*cs0 - 1i.*(Omega_S/2).*cp0;
cs0_prime_conj = conj(-1i.*Delta_S.*cs0_conj - 1i.*(Omega_S/2).*cp0_conj);

cd0_prime = -1i.*Delta_D.*cd0 - 1i.*(Omega_D/2).*cp0;
cd0_prime_conj = conj(-1i.*Delta_D.*cd0_conj - 1i.*(Omega_D/2).*cp0_conj);

cp0_prime = -1i.*(Omega_S/2).*cs0 - 1i.*(Omega_D/2).*cd0;
cp0_prime_conj = conj(-1i.*(Omega_S/2).*cs0_conj - 1i.*(Omega_D/2).*cd0_conj);

Evol_Matrix = nan(numel(cs0),numel(cs0)); %Initialize the matrix for 
%state evolution.

%Construct the time-evolution matrix for the density matrix.
Evol_Matrix(cs0.*cs0_conj == 1,:) = cs0.*cs0_prime_conj + cs0_prime.*cs0_conj...
    + gamma_S.*cp0.*cp0_conj;

Evol_Matrix(cd0.*cd0_conj == 1,:) = cd0.*cd0_prime_conj + cd0_prime.*cd0_conj...
    + gamma_D.*cp0.*cp0_conj;

Evol_Matrix(cp0.*cp0_conj == 1,:) = cp0.*cp0_prime_conj + cp0_prime.*cp0_conj...
    - (gamma_S + gamma_D).*cp0.*cp0_conj;

Evol_Matrix(cs0.*cd0_conj == 1,:) = cs0.*cd0_prime_conj + cs0_prime.*cd0_conj;
Evol_Matrix(cd0.*cs0_conj == 1,:) = cd0.*cs0_prime_conj + cd0_prime.*cs0_conj;

Evol_Matrix(cs0.*cp0_conj == 1,:) = cs0.*cp0_prime_conj + cs0_prime.*cp0_conj...
    - ((gamma_S + gamma_D)./2).*cs0.*cp0_conj;
Evol_Matrix(cp0.*cs0_conj == 1,:) = cp0.*cs0_prime_conj + cp0_prime.*cs0_conj...
    - ((gamma_S + gamma_D)./2).*cp0.*cs0_conj;

Evol_Matrix(cd0.*cp0_conj == 1,:) = cd0.*cp0_prime_conj + cd0_prime.*cp0_conj...
    - ((gamma_S + gamma_D)./2).*cd0.*cp0_conj;
Evol_Matrix(cp0.*cd0_conj == 1,:) = cp0.*cd0_prime_conj + cp0_prime.*cd0_conj...
    - ((gamma_S + gamma_D)./2).*cp0.*cd0_conj;

%Add another line to the Evol_Matrix to be used for constructing the restriction that the trace of the density matrix is 1.
Evol_Matrix(end+1,:) = zeros(1,9);
Evol_Matrix(end,cs0.*cs0_conj==1) = 1;
Evol_Matrix(end,cp0.*cp0_conj==1) = 1;
Evol_Matrix(end,cd0.*cd0_conj==1) = 1;

%Solve for the steady-state density matrix.
sigma_end = linsolve(Evol_Matrix,[zeros(9,1); 1]);