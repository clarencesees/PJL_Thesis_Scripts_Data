function [A,J_index,Jm_index,M1_index,M2_index]=GetClebschGordan(J1,J2)
%Input definition:
%J1: Angular momentum quantum number of first component
%J2: Angular momentum quantum number of second component
%Output definition:
%A: 2-dimensional matrix of size (2*J1+1)*(2*J2+1)x(2*J1+1)*(2*J2+1). The
%matrix containing all the Clebsch-Gordan coefficients. Rows correspond to
%the state |m1>|m2> while columns correspond to the state <j|<m|. v=A*u
%transforms the u vector in the |jm> basis to the |m1,m2> basis.
%J_index: 1-dimensional column vector of indeces of |j> state, ranges from
%J1+J2 to abs(J1-J2). e.g. J1=2, J2=1 => J_index = [3 3 3 3 3 3 3 2 2 2 2 2
%1 1 1]
%Jm_index: 1-dimensional column vector of indeces of |m> state. e.g. J1=2,
%J2=1 => Jm_index=[3 2 1 0 -1 -2 -3 2 1 0 -1 -2 1 0 -1];
%M1_index: 1-dimensioal row vector of z projection of angular momentum of
%first component. e.g. J1=2, J2=1 => M1_index = [2 2 2 1 1 1 0 0 0 -1 -1 -1
%-2 -2 -2]'
%M2_index: 1-dimensioal row vector of z projection of angular momentum of
%second component. e.g. J1=2, J2=1 => M1_index = [1 0 -1 1 0 -1 1 0 -1 1 0 -1
%1 0 -1]'
%
%To get the Clebsch-Gordan coefficient of a particular set of parameters,
%use A(M1_index==m1 & M2_index==m2,J_index==j & Jm_index==m) after running
%the function. e.g. we want m1=1, m2=1, j=3, m=2. 
%Type A(M1_index==1 & M2_index==1,J_index==3 & Jm_index==2).


J=J1+J2;
J_array=J:-1:abs(J1-J2);
J1_m=J1:-1:-J1;
J1_m=J1_m';
J2_m=J2:-1:-J2;
J2_m=J2_m';
%^Defines J, m1 and m2 states.

A=nan((2*J1+1)*(2*J2+1),(2*J1+1)*(2*J2+1)); %Initialize matrix for final results

J1_Lower=diag(sqrt(J1*(J1+1)-J1_m(1:end-1).*(J1_m(1:end-1)-1)),-1);
J2_Lower=diag(sqrt(J2*(J2+1)-J2_m(1:end-1).*(J2_m(1:end-1)-1)),-1);
J1_I=diag(ones(numel(J1_m),1));
J2_I=diag(ones(numel(J2_m),1));
%^Defines spin lowering and identity operators for first and second angular
%momentum components.

Jm_index=nan(1,(2*J1+1)*(2*J2+1));
J_index=nan(1,(2*J1+1)*(2*J2+1));
%^Initialize Jm_index and J_index defined in the output.

Jm_length=1;
for h=1:numel(J_array)
    Jm_index(Jm_length:2*J_array(h)+Jm_length)=J_array(h):-1:-J_array(h);
    J_index(Jm_length:2*J_array(h)+Jm_length)=J_array(h);
    Jm_length=Jm_length+2*J_array(h)+1;
end
%^Computes J_index and Jm_index.

M1_index=kron(J1_m,ones(numel(J2_m),1));
M2_index=kron(ones(numel(J1_m),1),J2_m);
%^Computes M1_index and M2_index.

Jm_length=1;
for hh=J:-1:abs(J1-J2)
    if hh==J %|m1=m1_max, m2=m2_max> = |j=j_max, m=m_max>
        v=zeros(numel(J1_m)*numel(J2_m),1);
        v(1)=1;
        A(:,1)=v;
    else %Compute transformation for the states |j=/=j_max, m=j> using 
        %simultaneous equation solver
        v=zeros(numel(J1_m)*numel(J2_m),1);
        V=A(:,J_index>hh & Jm_index==hh);
        V=V';
        V=V(:,M1_index+M2_index==hh);
        B=-V(:,1);
        V(:,1)=[];
        S=linsolve(V,B);
        S=[1; S]; %#ok<AGROW>
        v(M1_index+M2_index==hh)=S./norm(S);
        A(:,Jm_length)=v;
    end
    for h=Jm_length+1:2*hh+Jm_length %|j, m-1> = N*(J1_-+J2_-)|j,m>, N is 
        %a normalizing factor
        A(:,h)=((kron(J1_Lower,J2_I)+...
            kron(J1_I,J2_Lower))*A(:,h-1))./norm((kron(J1_Lower,J2_I)+...
            kron(J1_I,J2_Lower))*A(:,h-1));
    end
    Jm_length=Jm_length+2*hh+1;
end
%^Computes A.