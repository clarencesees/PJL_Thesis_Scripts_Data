Delta_S_array = -100:10:0; %Initialize cooling frequency scan array.
Delta_D_array = -200:10:200; %Initialize repump frequency scan array.
%Delta_S_array = -100:2:0; %Initialize cooling frequency scan array.
%Delta_D_array = -100:2:100; %Initialize repump frequency scan array.

%Ba137_SPDZeeman_Levels_OpticalBloch_init_prototype; %Initialize 

%Clebsch-Gordan and Wigner-Eckart coefficients.

%Initialize Rabi frequencies in MHz.
%Omega_S1P2 = 2*pi*10;
%Omega_S2P2 = 2*pi*10;
%Omega_D0P1 = 2*pi*10;
%Omega_D1P1 = 2*pi*10;
%Omega_D2P2 = 2*pi*10;
%Omega_D3P2 = 2*pi*10;

%Initialize polarizations.
%theta_s = 0;
%theta_d = 0;
%theta_s = theta;
%theta_d = theta;

%Initialize Lambda_detune.
%Lambda_detune = 10;

%Initialize 2D array of P state population for the frequency scan.
sigma_PP_end = nan(numel(Delta_S_array),numel(Delta_D_array)); 

for hh=1:numel(Delta_S_array)
    for hhh=1:numel(Delta_D_array)
        Delta_S1P2 = 2*pi*Delta_S_array(hh);
        Delta_S2P2 = 2*pi*(Delta_S_array(hh)+Lambda_detune);
        Delta_D0P1 = 2*pi*Delta_D_array(hhh);
        Delta_D1P1 = 2*pi*(Delta_D_array(hhh)+Lambda_detune);
        Delta_D2P2 = 2*pi*Delta_D_array(hhh);
        Delta_D3P2 = 2*pi*(Delta_D_array(hhh)+Lambda_detune);
        
        Ba137_SPDZeeman_Levels_OpticalBloch_linsolve_prototype;
        
        %Initialize CC, which stores the diagonal elements of the density
        %matrix.
        CC = nan(1,numel(s1_0));
        for EM_count=1:numel(s1_0)
            CC(EM_count) = sigma_end(c_array(EM_count,:).*c_conj_array(EM_count,:) == 1);
        end
        sigma_PP_end(hh,hhh) = sum(CC(25:end)); %Sum over all density 
        %matrix elements in the P states.
    end
end