Omega_S_array = 2:4:66; %Initialize S-P Rabi frequencies scan array in MHz.
Omega_D_array = 2:4:62; %Initialize D-P Rabi frequencies scan array in MHz.

%Initialize Rabi frequencies in MHz.
Omega_S = 50;
Omega_D = 50;

theta = pi/4; %Initialize polarizations.

Ba138_SPDZeeman_Levels_OpticalBloch_DeltaScan; %Perform frequency scan to get array size for initialization.

sigma_PP_end_O_array = nan(size(sigma_PP_end,1),size(sigma_PP_end,2),numel(Omega_S_array),numel(Omega_D_array)); %Initialize array for storing P state populations of frequency scan.
sigma_maxPP_O_array = nan(numel(Omega_S_array),numel(Omega_D_array)); %Initialize array for storing maximum P state population from frequency scan.

%Scan Rabi frequencies and get P state populations.
for Omega_S_counter = 1:numel(Omega_S_array)
    for Omega_D_counter = 1:numel(Omega_D_array)
        Omega_S = 2*pi*Omega_S_array(Omega_S_counter);
        Omega_D = 2*pi*Omega_D_array(Omega_D_counter);
        Ba138_SPDZeeman_Levels_OpticalBloch_DeltaScan;
        sigma_PP_end_O_array(:,:,Omega_S_counter,Omega_D_counter) = sigma_PP_end;
        sigma_maxPP_O_array(Omega_S_counter,Omega_D_counter) = max(sigma_PP_end(:));
    end
end