%This script initializes the Clebsch-Gordan and Wigner-Eckart coefficients 
%that is required for the script
%Ba137_SPDZeeman_Levels_OpticalBloch_linsolve_prototype.m. 
%
%A_XXYY_xxyy denote the Clebsch-Gordan coefficient for the transition from 
%the XX level with xx z-component angular momentum to the YY level with yy
%z-component angular momentum. For example, A_D0P1_01m is the
%Clebsch-Gordan coefficient for the transition from the D_{3/2} level with
%F = 0 and z-component angular momentum of m = 0 to the P_{1/2} level with
%F = 1 and z-component angular momentum of m = -1. The glossary is as
%follows:
%For the hyperfine levels:
%S[X] denotes the S_{1/2} level with F = [X].
%P[X] denotes the P_{1/2} level with F = [X].
%D[X] denotes the D_{3/2} level with F = [X].
%For the Zeeman levels:
%m denotes minus. e.g. [x]m denotes the Zeeman level m = -[x].
%p denotes plus. e.g. [x]p denotes the Zeeman level m = [x].
%0 denotes m = 0.
%
%W_XXYY_xxyy denote the Wigner-Eckart coefficient for the transition from
%the XX level with xx z-component angular momentum to the YY level with yy
%z-component angular momentum. For example, W_D0P1_01m is the
%Wigner-Eckart coefficient for the transition from the D_{3/2} level with
%F = 0 and z-component angular momentum of m = 0 to the P_{1/2} level with
%F = 1 and z-component angular momentum of m = -1.
%
%In addition to computing the Clebsch-Gordan and Wigner-Eckart
%coefficients, the script also initializes the Rabi frequency for each
%transition to be zero. Similarly, the detuning for each laser transition
%is also initialized to be zero.
%Omega_XXYY denotes the Rabi frequency from the XX level to YY level.
%Delta_XXYY denotes the laser detuning from the XX level to YY level.
%
%This script requires the scripts GetClebschGordan.m FmtoJCoefficient.m and
%NotWigner6j.m in the working folder to run.

[A,J_index,Jm_index,M1_index,M2_index]=GetClebschGordan(0,1); %Generates 
%the Clebsch-Gordan matrix A with J1 = 0 and J2 = 1, the specific 
%coefficient is searched using the J, Jm, M1 and M2 indeces.
A(abs(A)<10^(-10)) = 0; %This line gets rid of machine round off error for 
%zeroes.
%The following for loop corrects for non-zero round off error using
%MATLAB's rat module.
for A_ind = 1:numel(A)
    A(A_ind) = sign(A(A_ind)).*sqrt(str2num(rat(A(A_ind)^2)));
end

%The following 3 lines assign the Clebsch-Gordan coefficients from the A
%matrix to all the D0 to P1 transitions.
A_D0P1_01m = A(M1_index==0 & M2_index==-1,J_index==1 & Jm_index==-1);
A_D0P1_00 = A(M1_index==0 & M2_index==0,J_index==1 & Jm_index==0);
A_D0P1_01p = A(M1_index==0 & M2_index==1,J_index==1 & Jm_index==1);


[A,J_index,Jm_index,M1_index,M2_index]=GetClebschGordan(1,1); %Generates 
%the Clebsch-Gordan matrix A with J1 = 1 and J2 = 1, the specific 
%coefficient is searched using the J, Jm, M1 and M2 indeces.
A(abs(A)<10^(-10)) = 0; %This line gets rid of machine round off error for 
%zeroes.
%The following for loop corrects for non-zero round off error using
%MATLAB's rat module.
for A_ind = 1:numel(A)
    A(A_ind) = sign(A(A_ind)).*sqrt(str2num(rat(A(A_ind)^2)));
end

%The lines 69 to 105 assign the Clebsch-Gordan coefficients from the A
%matrix to all the transitions with F = 1 as the starting level.
A_S1P1_1m1m = A(M1_index==-1 & M2_index==0,J_index==1 & Jm_index==-1);
A_S1P1_1m0 = A(M1_index==-1 & M2_index==1,J_index==1 & Jm_index==0);
A_D1P1_1m1m = A(M1_index==-1 & M2_index==0,J_index==1 & Jm_index==-1);
A_D1P1_1m0 = A(M1_index==-1 & M2_index==1,J_index==1 & Jm_index==0);

A_S1P1_01m = A(M1_index==0 & M2_index==-1,J_index==1 & Jm_index==-1);
A_S1P1_00 = A(M1_index==0 & M2_index==0,J_index==1 & Jm_index==0);
A_S1P1_01p = A(M1_index==0 & M2_index==1,J_index==1 & Jm_index==1);
A_D1P1_01m = A(M1_index==0 & M2_index==-1,J_index==1 & Jm_index==-1);
A_D1P1_00 = A(M1_index==0 & M2_index==0,J_index==1 & Jm_index==0);
A_D1P1_01p = A(M1_index==0 & M2_index==1,J_index==1 & Jm_index==1);

A_S1P1_1p0 = A(M1_index==1 & M2_index==-1,J_index==1 & Jm_index==0);
A_S1P1_1p1p = A(M1_index==1 & M2_index==0,J_index==1 & Jm_index==1);
A_D1P1_1p0 = A(M1_index==1 & M2_index==-1,J_index==1 & Jm_index==0);
A_D1P1_1p1p = A(M1_index==1 & M2_index==0,J_index==1 & Jm_index==1);

A_S1P2_1m2m = A(M1_index==-1 & M2_index==-1,J_index==2 & Jm_index==-2);
A_S1P2_1m1m = A(M1_index==-1 & M2_index==0,J_index==2 & Jm_index==-1);
A_S1P2_1m0 = A(M1_index==-1 & M2_index==1,J_index==2 & Jm_index==0);
A_D1P2_1m2m = A(M1_index==-1 & M2_index==-1,J_index==2 & Jm_index==-2);
A_D1P2_1m1m = A(M1_index==-1 & M2_index==0,J_index==2 & Jm_index==-1);
A_D1P2_1m0 = A(M1_index==-1 & M2_index==1,J_index==2 & Jm_index==0);

A_S1P2_01m = A(M1_index==0 & M2_index==-1,J_index==2 & Jm_index==-1);
A_S1P2_00 = A(M1_index==0 & M2_index==0,J_index==2 & Jm_index==0);
A_S1P2_01p = A(M1_index==0 & M2_index==1,J_index==2 & Jm_index==1);
A_D1P2_01m = A(M1_index==0 & M2_index==-1,J_index==2 & Jm_index==-1);
A_D1P2_00 = A(M1_index==0 & M2_index==0,J_index==2 & Jm_index==0);
A_D1P2_01p = A(M1_index==0 & M2_index==1,J_index==2 & Jm_index==1);

A_S1P2_1p0 = A(M1_index==1 & M2_index==-1,J_index==2 & Jm_index==0);
A_S1P2_1p1p = A(M1_index==1 & M2_index==0,J_index==2 & Jm_index==1);
A_S1P2_1p2p = A(M1_index==1 & M2_index==1,J_index==2 & Jm_index==2);
A_D1P2_1p0 = A(M1_index==1 & M2_index==-1,J_index==2 & Jm_index==0);
A_D1P2_1p1p = A(M1_index==1 & M2_index==0,J_index==2 & Jm_index==1);
A_D1P2_1p2p = A(M1_index==1 & M2_index==1,J_index==2 & Jm_index==2);


[A,J_index,Jm_index,M1_index,M2_index]=GetClebschGordan(2,1); %Generates 
%the Clebsch-Gordan matrix A with J1 = 2 and J2 = 1, the specific 
%coefficient is searched using the J, Jm, M1 and M2 indeces.
A(abs(A)<10^(-10)) = 0; %This line gets rid of machine round off error for 
%zeroes.
%The following for loop corrects for non-zero round off error using
%MATLAB's rat module.
for A_ind = 1:numel(A)
    A(A_ind) = sign(A(A_ind)).*sqrt(str2num(rat(A(A_ind)^2)));
end

%The lines 121 to 173 assign the Clebsch-Gordan coefficients from the A
%matrix to all the transitions with F = 2 as the starting level.
A_S2P1_2m1m = A(M1_index==-2 & M2_index==1,J_index==1 & Jm_index==-1);
A_D2P1_2m1m = A(M1_index==-2 & M2_index==1,J_index==1 & Jm_index==-1);

A_S2P1_1m1m = A(M1_index==-1 & M2_index==0,J_index==1 & Jm_index==-1);
A_S2P1_1m0 = A(M1_index==-1 & M2_index==1,J_index==1 & Jm_index==0);
A_D2P1_1m1m = A(M1_index==-1 & M2_index==0,J_index==1 & Jm_index==-1);
A_D2P1_1m0 = A(M1_index==-1 & M2_index==1,J_index==1 & Jm_index==0);

A_S2P1_01m = A(M1_index==0 & M2_index==-1,J_index==1 & Jm_index==-1);
A_S2P1_00 = A(M1_index==0 & M2_index==0,J_index==1 & Jm_index==0);
A_S2P1_01p = A(M1_index==0 & M2_index==1,J_index==1 & Jm_index==1);
A_D2P1_01m = A(M1_index==0 & M2_index==-1,J_index==1 & Jm_index==-1);
A_D2P1_00 = A(M1_index==0 & M2_index==0,J_index==1 & Jm_index==0);
A_D2P1_01p = A(M1_index==0 & M2_index==1,J_index==1 & Jm_index==1);

A_S2P1_1p0 = A(M1_index==1 & M2_index==-1,J_index==1 & Jm_index==0);
A_S2P1_1p1p = A(M1_index==1 & M2_index==0,J_index==1 & Jm_index==1);
A_D2P1_1p0 = A(M1_index==1 & M2_index==-1,J_index==1 & Jm_index==0);
A_D2P1_1p1p = A(M1_index==1 & M2_index==0,J_index==1 & Jm_index==1);

A_S2P1_2p1p = A(M1_index==2 & M2_index==-1,J_index==1 & Jm_index==1);
A_D2P1_2p1p = A(M1_index==2 & M2_index==-1,J_index==1 & Jm_index==1);

A_S2P2_2m2m = A(M1_index==-2 & M2_index==0,J_index==2 & Jm_index==-2);
A_S2P2_2m1m = A(M1_index==-2 & M2_index==1,J_index==2 & Jm_index==-1);
A_D2P2_2m2m = A(M1_index==-2 & M2_index==0,J_index==2 & Jm_index==-2);
A_D2P2_2m1m = A(M1_index==-2 & M2_index==1,J_index==2 & Jm_index==-1);

A_S2P2_1m2m = A(M1_index==-1 & M2_index==-1,J_index==2 & Jm_index==-2);
A_S2P2_1m1m = A(M1_index==-1 & M2_index==0,J_index==2 & Jm_index==-1);
A_S2P2_1m0 = A(M1_index==-1 & M2_index==1,J_index==2 & Jm_index==0);
A_D2P2_1m2m = A(M1_index==-1 & M2_index==-1,J_index==2 & Jm_index==-2);
A_D2P2_1m1m = A(M1_index==-1 & M2_index==0,J_index==2 & Jm_index==-1);
A_D2P2_1m0 = A(M1_index==-1 & M2_index==1,J_index==2 & Jm_index==0);

A_S2P2_01m = A(M1_index==0 & M2_index==-1,J_index==2 & Jm_index==-1);
A_S2P2_00 = A(M1_index==0 & M2_index==0,J_index==2 & Jm_index==0);
A_S2P2_01p = A(M1_index==0 & M2_index==1,J_index==2 & Jm_index==1);
A_D2P2_01m = A(M1_index==0 & M2_index==-1,J_index==2 & Jm_index==-1);
A_D2P2_00 = A(M1_index==0 & M2_index==0,J_index==2 & Jm_index==0);
A_D2P2_01p = A(M1_index==0 & M2_index==1,J_index==2 & Jm_index==1);

A_S2P2_1p0 = A(M1_index==1 & M2_index==-1,J_index==2 & Jm_index==0);
A_S2P2_1p1p = A(M1_index==1 & M2_index==0,J_index==2 & Jm_index==1);
A_S2P2_1p2p = A(M1_index==1 & M2_index==1,J_index==2 & Jm_index==2);
A_D2P2_1p0 = A(M1_index==1 & M2_index==-1,J_index==2 & Jm_index==0);
A_D2P2_1p1p = A(M1_index==1 & M2_index==0,J_index==2 & Jm_index==1);
A_D2P2_1p2p = A(M1_index==1 & M2_index==1,J_index==2 & Jm_index==2);

A_S2P2_2p1p = A(M1_index==2 & M2_index==-1,J_index==2 & Jm_index==1);
A_S2P2_2p2p = A(M1_index==2 & M2_index==0,J_index==2 & Jm_index==2);
A_D2P2_2p1p = A(M1_index==2 & M2_index==-1,J_index==2 & Jm_index==1);
A_D2P2_2p2p = A(M1_index==2 & M2_index==0,J_index==2 & Jm_index==2);


[A,J_index,Jm_index,M1_index,M2_index]=GetClebschGordan(3,1); %Generates 
%the Clebsch-Gordan matrix A with J1 = 3 and J2 = 1, the specific 
%coefficient is searched using the J, Jm, M1 and M2 indeces.
A(abs(A)<10^(-10)) = 0; %This line gets rid of machine round off error for 
%zeroes.
%The following for loop corrects for non-zero round off error using
%MATLAB's rat module.
for A_ind = 1:numel(A)
    A(A_ind) = sign(A(A_ind)).*sqrt(str2num(rat(A(A_ind)^2)));
end

%The lines 189 to 209 assign the Clebsch-Gordan coefficients from the A
%matrix to all the transitions with F = 3 as the starting level.
A_D3P2_3m2m = A(M1_index==-3 & M2_index==1,J_index==2 & Jm_index==-2);

A_D3P2_2m2m = A(M1_index==-2 & M2_index==0,J_index==2 & Jm_index==-2);
A_D3P2_2m1m = A(M1_index==-2 & M2_index==1,J_index==2 & Jm_index==-1);

A_D3P2_1m2m = A(M1_index==-1 & M2_index==-1,J_index==2 & Jm_index==-2);
A_D3P2_1m1m = A(M1_index==-1 & M2_index==0,J_index==2 & Jm_index==-1);
A_D3P2_1m0 = A(M1_index==-1 & M2_index==1,J_index==2 & Jm_index==0);

A_D3P2_01m = A(M1_index==0 & M2_index==-1,J_index==2 & Jm_index==-1);
A_D3P2_00 = A(M1_index==0 & M2_index==0,J_index==2 & Jm_index==0);
A_D3P2_01p = A(M1_index==0 & M2_index==1,J_index==2 & Jm_index==1);

A_D3P2_1p0 = A(M1_index==1 & M2_index==-1,J_index==2 & Jm_index==0);
A_D3P2_1p1p = A(M1_index==1 & M2_index==0,J_index==2 & Jm_index==1);
A_D3P2_1p2p = A(M1_index==1 & M2_index==1,J_index==2 & Jm_index==2);

A_D3P2_2p1p = A(M1_index==2 & M2_index==-1,J_index==2 & Jm_index==1);
A_D3P2_2p2p = A(M1_index==2 & M2_index==0,J_index==2 & Jm_index==2);

A_D3P2_3p2p = A(M1_index==3 & M2_index==-1,J_index==2 & Jm_index==2);

%Lines 213 to 220 initializes the hyperfine level values for computing
%Wigner-Eckart coefficients.
S1 = 1;
S2 = 2;
D0 = 0;
D1 = 1;
D2 = 2;
D3 = 3;
P1 = 1;
P2 = 2;

%Lines 224 to 230 initializes the Zeeman level values for computing
%Wigner-Eckart coefficients.
m3 = -3;
m2 = -2;
m1 = -1;
p0 = 0;
p1 = 1;
p2 = 2;
p3 = 3;

%Lines 234 to 237 initializes the J and I angular momentum values for
%computing Wigner-Eckart coefficients.
S_J = 1/2;
D_J = 3/2;
P_J = 1/2;
I_spin = 3/2;

%Lines 241 to 373 compute the Wigner-Eckart coefficient for each
%transition. The same round-off correction method using rat is used.
W_S1P1_1m1m = str2num(rat(FmtoJCoefficient(P1,m1,S1,m1,1,m1-m1,P_J,S_J,I_spin).^2));
W_S1P1_1m0 = str2num(rat(FmtoJCoefficient(P1,0,S1,m1,1,0-m1,P_J,S_J,I_spin).^2));

W_S1P1_01m = str2num(rat(FmtoJCoefficient(P1,m1,S1,0,1,m1-0,P_J,S_J,I_spin).^2));
W_S1P1_00 = str2num(rat(FmtoJCoefficient(P1,0,S1,0,1,0-0,P_J,S_J,I_spin).^2));
W_S1P1_01p = str2num(rat(FmtoJCoefficient(P1,p1,S1,0,1,p1-0,P_J,S_J,I_spin).^2));

W_S1P1_1p0 = str2num(rat(FmtoJCoefficient(P1,0,S1,p1,1,0-p1,P_J,S_J,I_spin).^2));
W_S1P1_1p1p = str2num(rat(FmtoJCoefficient(P1,p1,S1,p1,1,p1-p1,P_J,S_J,I_spin).^2));

W_S1P2_1m2m = str2num(rat(FmtoJCoefficient(P2,m2,S1,m1,1,m2-m1,P_J,S_J,I_spin).^2));
W_S1P2_1m1m = str2num(rat(FmtoJCoefficient(P2,m1,S1,m1,1,m1-m1,P_J,S_J,I_spin).^2));
W_S1P2_1m0 = str2num(rat(FmtoJCoefficient(P2,0,S1,m1,1,0-m1,P_J,S_J,I_spin).^2));

W_S1P2_01m = str2num(rat(FmtoJCoefficient(P2,m1,S1,0,1,m1-0,P_J,S_J,I_spin).^2));
W_S1P2_00 = str2num(rat(FmtoJCoefficient(P2,0,S1,0,1,0-0,P_J,S_J,I_spin).^2));
W_S1P2_01p = str2num(rat(FmtoJCoefficient(P2,p1,S1,0,1,p1-0,P_J,S_J,I_spin).^2));

W_S1P2_1p0 = str2num(rat(FmtoJCoefficient(P2,0,S1,p1,1,0-p1,P_J,S_J,I_spin).^2));
W_S1P2_1p1p = str2num(rat(FmtoJCoefficient(P2,p1,S1,p1,1,p1-p1,P_J,S_J,I_spin).^2));
W_S1P2_1p2p = str2num(rat(FmtoJCoefficient(P2,p2,S1,p1,1,p2-p1,P_J,S_J,I_spin).^2));

W_S2P1_2m1m = str2num(rat(FmtoJCoefficient(P1,m1,S2,m2,1,m1-m2,P_J,S_J,I_spin).^2));

W_S2P1_1m1m = str2num(rat(FmtoJCoefficient(P1,m1,S2,m1,1,m1-m1,P_J,S_J,I_spin).^2));
W_S2P1_1m0 = str2num(rat(FmtoJCoefficient(P1,0,S2,m1,1,0-m1,P_J,S_J,I_spin).^2));

W_S2P1_01m = str2num(rat(FmtoJCoefficient(P1,m1,S2,0,1,m1-0,P_J,S_J,I_spin).^2));
W_S2P1_00 = str2num(rat(FmtoJCoefficient(P1,0,S2,0,1,0-0,P_J,S_J,I_spin).^2));
W_S2P1_01p = str2num(rat(FmtoJCoefficient(P1,p1,S2,0,1,p1-0,P_J,S_J,I_spin).^2));

W_S2P1_1p0 = str2num(rat(FmtoJCoefficient(P1,0,S2,p1,1,0-p1,P_J,S_J,I_spin).^2));
W_S2P1_1p1p = str2num(rat(FmtoJCoefficient(P1,p1,S2,p1,1,p1-p1,P_J,S_J,I_spin).^2));

W_S2P1_2p1p = str2num(rat(FmtoJCoefficient(P1,p1,S2,p2,1,p1-p2,P_J,S_J,I_spin).^2));

W_S2P2_2m2m = str2num(rat(FmtoJCoefficient(P2,m2,S2,m2,1,m2-m2,P_J,S_J,I_spin).^2));
W_S2P2_2m1m = str2num(rat(FmtoJCoefficient(P2,m1,S2,m2,1,m1-m2,P_J,S_J,I_spin).^2));

W_S2P2_1m2m = str2num(rat(FmtoJCoefficient(P2,m2,S2,m1,1,m2-m1,P_J,S_J,I_spin).^2));
W_S2P2_1m1m = str2num(rat(FmtoJCoefficient(P2,m1,S2,m1,1,m1-m1,P_J,S_J,I_spin).^2));
W_S2P2_1m0 = str2num(rat(FmtoJCoefficient(P2,0,S2,m1,1,0-m1,P_J,S_J,I_spin).^2));

W_S2P2_01m = str2num(rat(FmtoJCoefficient(P2,m1,S2,0,1,m1-0,P_J,S_J,I_spin).^2));
W_S2P2_00 = str2num(rat(FmtoJCoefficient(P2,0,S2,0,1,0-0,P_J,S_J,I_spin).^2));
W_S2P2_01p = str2num(rat(FmtoJCoefficient(P2,p1,S2,0,1,p1-0,P_J,S_J,I_spin).^2));

W_S2P2_1p0 = str2num(rat(FmtoJCoefficient(P2,0,S2,p1,1,0-p1,P_J,S_J,I_spin).^2));
W_S2P2_1p1p = str2num(rat(FmtoJCoefficient(P2,p1,S2,p1,1,p1-p1,P_J,S_J,I_spin).^2));
W_S2P2_1p2p = str2num(rat(FmtoJCoefficient(P2,p2,S2,p1,1,p2-p1,P_J,S_J,I_spin).^2));

W_S2P2_2p1p = str2num(rat(FmtoJCoefficient(P2,p1,S2,p2,1,p1-p2,P_J,S_J,I_spin).^2));
W_S2P2_2p2p = str2num(rat(FmtoJCoefficient(P2,p2,S2,p2,1,p2-p2,P_J,S_J,I_spin).^2));

W_D0P1_01m = str2num(rat(FmtoJCoefficient(P1,m1,D0,0,1,m1-0,P_J,D_J,I_spin).^2));
W_D0P1_00 = str2num(rat(FmtoJCoefficient(P1,0,D0,0,1,0-0,P_J,D_J,I_spin).^2));
W_D0P1_01p = str2num(rat(FmtoJCoefficient(P1,p1,D0,0,1,p1-0,P_J,D_J,I_spin).^2));

W_D1P1_1m1m = str2num(rat(FmtoJCoefficient(P1,m1,D1,0,1,m1-m1,P_J,D_J,I_spin).^2));
W_D1P1_1m0 = str2num(rat(FmtoJCoefficient(P1,0,D1,m1,1,0-m1,P_J,D_J,I_spin).^2));

W_D1P1_01m = str2num(rat(FmtoJCoefficient(P1,m1,D1,0,1,m1-0,P_J,D_J,I_spin).^2));
W_D1P1_00 = str2num(rat(FmtoJCoefficient(P1,0,D1,0,1,0-0,P_J,D_J,I_spin).^2));
W_D1P1_01p = str2num(rat(FmtoJCoefficient(P1,p1,D1,0,1,p1-0,P_J,D_J,I_spin).^2));

W_D1P1_1p0 = str2num(rat(FmtoJCoefficient(P1,0,D1,p1,1,0-p1,P_J,D_J,I_spin).^2));
W_D1P1_1p1p = str2num(rat(FmtoJCoefficient(P1,p1,D1,p1,1,p1-p1,P_J,D_J,I_spin).^2));

W_D1P2_1m2m = str2num(rat(FmtoJCoefficient(P2,m2,D1,m1,1,m2-m1,P_J,D_J,I_spin).^2));
W_D1P2_1m1m = str2num(rat(FmtoJCoefficient(P2,m1,D1,m1,1,m1-m1,P_J,D_J,I_spin).^2));
W_D1P2_1m0 = str2num(rat(FmtoJCoefficient(P2,0,D1,m1,1,0-m1,P_J,D_J,I_spin).^2));

W_D1P2_01m = str2num(rat(FmtoJCoefficient(P2,m1,D1,0,1,m1-0,P_J,D_J,I_spin).^2));
W_D1P2_00 = str2num(rat(FmtoJCoefficient(P2,0,D1,0,1,0-0,P_J,D_J,I_spin).^2));
W_D1P2_01p = str2num(rat(FmtoJCoefficient(P2,p1,D1,0,1,p1-0,P_J,D_J,I_spin).^2));

W_D1P2_1p0 = str2num(rat(FmtoJCoefficient(P2,0,D1,p1,1,0-p1,P_J,D_J,I_spin).^2));
W_D1P2_1p1p = str2num(rat(FmtoJCoefficient(P2,p1,D1,p1,1,p1-p1,P_J,D_J,I_spin).^2));
W_D1P2_1p2p = str2num(rat(FmtoJCoefficient(P2,p2,D1,p1,1,p2-p1,P_J,D_J,I_spin).^2));

W_D2P1_2m1m = str2num(rat(FmtoJCoefficient(P1,m1,D2,m2,1,m1-m2,P_J,D_J,I_spin).^2));

W_D2P1_1m1m = str2num(rat(FmtoJCoefficient(P1,m1,D2,m1,1,m1-m1,P_J,D_J,I_spin).^2));
W_D2P1_1m0 = str2num(rat(FmtoJCoefficient(P1,0,D2,m1,1,0-m1,P_J,D_J,I_spin).^2));

W_D2P1_01m = str2num(rat(FmtoJCoefficient(P1,m1,D2,0,1,m1-0,P_J,D_J,I_spin).^2));
W_D2P1_00 = str2num(rat(FmtoJCoefficient(P1,0,D2,0,1,0-0,P_J,D_J,I_spin).^2));
W_D2P1_01p = str2num(rat(FmtoJCoefficient(P1,p1,D2,0,1,p1-0,P_J,D_J,I_spin).^2));

W_D2P1_1p0 = str2num(rat(FmtoJCoefficient(P1,0,D2,p1,1,0-p1,P_J,D_J,I_spin).^2));
W_D2P1_1p1p = str2num(rat(FmtoJCoefficient(P1,p1,D2,p1,1,p1-p1,P_J,D_J,I_spin).^2));

W_D2P1_2p1p = str2num(rat(FmtoJCoefficient(P1,p1,D2,p2,1,p1-p2,P_J,D_J,I_spin).^2));

W_D2P2_2m2m = str2num(rat(FmtoJCoefficient(P2,m2,D2,m2,1,m2-m2,P_J,D_J,I_spin).^2));
W_D2P2_2m1m = str2num(rat(FmtoJCoefficient(P2,m1,D2,m2,1,m1-m2,P_J,D_J,I_spin).^2));

W_D2P2_1m2m = str2num(rat(FmtoJCoefficient(P2,m2,D2,m1,1,m2-m1,P_J,D_J,I_spin).^2));
W_D2P2_1m1m = str2num(rat(FmtoJCoefficient(P2,m1,D2,m1,1,m1-m1,P_J,D_J,I_spin).^2));
W_D2P2_1m0 = str2num(rat(FmtoJCoefficient(P2,0,D2,m1,1,0-m1,P_J,D_J,I_spin).^2));

W_D2P2_01m = str2num(rat(FmtoJCoefficient(P2,m1,D2,0,1,m1-0,P_J,D_J,I_spin).^2));
W_D2P2_00 = str2num(rat(FmtoJCoefficient(P2,0,D2,0,1,0-0,P_J,D_J,I_spin).^2));
W_D2P2_01p = str2num(rat(FmtoJCoefficient(P2,p1,D2,0,1,p1-0,P_J,D_J,I_spin).^2));

W_D2P2_1p0 = str2num(rat(FmtoJCoefficient(P2,0,D2,p1,1,0-p1,P_J,D_J,I_spin).^2));
W_D2P2_1p1p = str2num(rat(FmtoJCoefficient(P2,p1,D2,p1,1,p1-p1,P_J,D_J,I_spin).^2));
W_D2P2_1p2p = str2num(rat(FmtoJCoefficient(P2,p2,D2,p1,1,p2-p1,P_J,D_J,I_spin).^2));

W_D2P2_2p1p = str2num(rat(FmtoJCoefficient(P2,p1,D2,p2,1,p1-p2,P_J,D_J,I_spin).^2));
W_D2P2_2p2p = str2num(rat(FmtoJCoefficient(P2,p2,D2,p2,1,p2-p2,P_J,D_J,I_spin).^2));

W_D3P2_3m2m = str2num(rat(FmtoJCoefficient(P2,m2,D3,m3,1,m2-m3,P_J,D_J,I_spin).^2));

W_D3P2_2m2m = str2num(rat(FmtoJCoefficient(P2,m2,D3,m2,1,m2-m2,P_J,D_J,I_spin).^2));
W_D3P2_2m1m = str2num(rat(FmtoJCoefficient(P2,m1,D3,m2,1,m1-m2,P_J,D_J,I_spin).^2));

W_D3P2_1m2m = str2num(rat(FmtoJCoefficient(P2,m2,D3,m1,1,m2-m1,P_J,D_J,I_spin).^2));
W_D3P2_1m1m = str2num(rat(FmtoJCoefficient(P2,m1,D3,m1,1,m1-m1,P_J,D_J,I_spin).^2));
W_D3P2_1m0 = str2num(rat(FmtoJCoefficient(P2,0,D3,m1,1,0-m1,P_J,D_J,I_spin).^2));

W_D3P2_01m = str2num(rat(FmtoJCoefficient(P2,m1,D3,0,1,m1-0,P_J,D_J,I_spin).^2));
W_D3P2_00 = str2num(rat(FmtoJCoefficient(P2,0,D3,0,1,0-0,P_J,D_J,I_spin).^2));
W_D3P2_01p = str2num(rat(FmtoJCoefficient(P2,p1,D3,0,1,p1-0,P_J,D_J,I_spin).^2));

W_D3P2_1p0 = str2num(rat(FmtoJCoefficient(P2,0,D3,p1,1,0-p1,P_J,D_J,I_spin).^2));
W_D3P2_1p1p = str2num(rat(FmtoJCoefficient(P2,p1,D3,p1,1,p1-p1,P_J,D_J,I_spin).^2));
W_D3P2_1p2p = str2num(rat(FmtoJCoefficient(P2,p2,D3,p1,1,p2-p1,P_J,D_J,I_spin).^2));

W_D3P2_2p1p = str2num(rat(FmtoJCoefficient(P2,p1,D3,p2,1,p1-p2,P_J,D_J,I_spin).^2));
W_D3P2_2p2p = str2num(rat(FmtoJCoefficient(P2,p2,D3,p2,1,p2-p2,P_J,D_J,I_spin).^2));

W_D3P2_3p2p = str2num(rat(FmtoJCoefficient(P2,p2,D3,p3,1,p2-p3,P_J,D_J,I_spin).^2));

%Lines 377 to 386 initializes the Rabi frequencies and laser detunings to
%zero for all transitions.
Omega_S1P1 = 0; Delta_S1P1 = 0;
Omega_S1P2 = 0; Delta_S1P2 = 0;
Omega_S2P1 = 0; Delta_S2P1 = 0;
Omega_S2P2 = 0; Delta_S2P2 = 0;
Omega_D0P1 = 0; Delta_D0P1 = 0;
Omega_D1P1 = 0; Delta_D1P1 = 0;
Omega_D1P2 = 0; Delta_D1P2 = 0;
Omega_D2P1 = 0; Delta_D2P1 = 0;
Omega_D2P2 = 0; Delta_D2P2 = 0;
Omega_D3P2 = 0; Delta_D3P2 = 0;