function result=FmtoJCoefficient(F1,m1,F2,m2,k,q,J1,J2,I)
%FmtoLCoefficient(F1,m1,F2,m2,k,q,J1,J2,I,L1,L2,S)
%
%Output definition:
%result: Coefficient that reduces transition matrix element from |F,m_F> to
%|J>. Scalar.
%
%Input definition:
%F1: F level of state 1. Scalar.
%m1: z-angular momentum of state 1. Scalar.
%F2: F level of state 2. Scalar.
%m2: z-angular momentum of state 2. Scalar.
%k: Tensor rank. 1 for dipole transition. Scalar.
%q: Polarization of perturbation. Scalar.
%J1: J level of state 1. Scalar.
%J2: J level of state 2. Scalar.
%I: Nuclear spin. Scalar.

[A,J_index,Jm_index,M1_index,M2_index]=GetClebschGordan(F2,k); %Generate
%the Clebsch-Gordan matrix
C1=A(M1_index==m2 & M2_index==q,J_index==F1 & Jm_index==m1); %Pick the
%Clebsch-Gordan coefficient corresponding to coupling of (m2, q) to 
%(F1, m1)

if isempty(C1)
    C1=0;
end
result=C1;

%Reduce the transition matrix element to J form
result=result*NotWigner6j(F1,F2,J1,J2,I,k);

%Reduce the transition matrix element to L form
%result=result*NotWigner6j(J1,J2,L1,L2,S,k);