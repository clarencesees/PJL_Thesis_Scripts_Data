function result=NotWigner6j(J,Jp,J1,J1p,J2,k)
%Output Definition:
%result: The coefficient obtained after multiplying the Wigner-6j term with
%the factor (-1)^(Jp+J1+k+J2)*sqrt((2*Jp+1)(2*J1+1)). Scalar

%Input Definition:
%J: Unreduced angular momentum of state 1. Scalar.
%Jp: Unreduced angular momentum of state 2. Scalar.
%J1: Reduced angular momentum of state 1. Scalar.
%J1p: Reduced angular momentum of state 2. Scalar.
%J2: Angular momentum component that combines with J1 to form J. Scalar
%k: Rank of tensor operator. Scalar

%*Refer to https://ions-wiki.iqc.uwaterloo.ca/index.php?title=Rabi_Frequency-Laser_Intensity_Relation
%for details.

m=J; %Computed coefficient is independent of z-angular momentum of unreduced
%angular momentum. Just need to choose 1 and set m=J for convenience.

%Generate Clebsch-Gordan Matrices needed to compute the coefficients.
[A1,A1_J_index,A1_Jm_index,A1_M1_index,A1_M2_index]=GetClebschGordan(J1,J2);
[A2,A2_Jp_index,A2_Jpm_index,A2_M1p_index,A2_M2_index]=GetClebschGordan(J1p,J2);
[A3,A3_J_index,A3_Jm_index,A3_Mp_index,A3_q_index]=GetClebschGordan(Jp,k);
[A4,A4_J1_index,A4_J1m_index,A4_M1p_index,A4_q_index]=GetClebschGordan(J1p,k);

result=0; %Initialize result
%Compute the summation
for mp=-Jp:1:Jp
    for q=-k:1:k
        for m1=-J1:1:J1
            for m2=-J2:1:J2
                for m1p=-J1p:1:J1p
                    C1=A1(A1_M1_index==m1 & A1_M2_index==m2,A1_J_index==J & A1_Jm_index==m);
                    C2=A2(A2_M1p_index==m1p & A2_M2_index==m2,A2_Jp_index==Jp & A2_Jpm_index==mp);
                    C3=A3(A3_Mp_index==mp & A3_q_index==q,A3_J_index==J & A3_Jm_index==m);
                    C4=A4(A4_M1p_index==m1p & A4_q_index==q,A4_J1_index==J1 & A4_J1m_index==m1);
                    if isempty(C1)
                        C1=0;
                    elseif isempty(C2)
                        C2=0;
                    elseif isempty(C3)
                        C3=0;
                    elseif isempty(C4)
                        C4=0;
                    end
                    result=result+C1*C2*C3*C4;
                end
            end
        end
    end
end