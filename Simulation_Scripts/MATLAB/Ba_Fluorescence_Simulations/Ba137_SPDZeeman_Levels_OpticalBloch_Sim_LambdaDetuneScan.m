Lambda_detune_large_array = 10:1:100; %Initialize relative frequency detuning scan array.

Ba137_SPDZeeman_Levels_OpticalBloch_init_prototype; %Initialize Clebsch-Gordan and Wigner-Eckart coefficients.

%Initialize Rabi frequencies in MHz.
Omega_S1P2 = 100;
Omega_S2P2 = 100;
Omega_D0P1 = 100;
Omega_D1P1 = 100;
Omega_D2P2 = 100;
Omega_D3P2 = 100;

theta = 0; %Initialize polarizations.

Lambda_detune = 1;

Ba137_SPDZeeman_Levels_OpticalBloch_DeltaScan; %Perform frequency scan to get array size for initialization.

sigma_PP_end_detune_large_array = nan(size(sigma_PP_end,1),size(sigma_PP_end,2),numel(Lambda_detune_large_array)); %Initialize array for storing P state populations of frequency scan.
sigma_maxPP_detune_large_array = nan(numel(1,Lambda_detune_large_array)); %Initialize array for storing maximum P state population from frequency scan.

%Scan relative frequency detuning and get P state populations.
for Lambda_detune_counter = 1:numel(Lambda_detune_large_array)
    Lambda_detune = Lambda_detune_large_array(Lambda_detune_counter);
    Ba137_SPDZeeman_Levels_OpticalBloch_DeltaScan;
    sigma_PP_end_detune_large_array(:,:,Lambda_detune_counter);
    sigma_maxPP_detune_large_array(Lambda_detune_counter) = max(sigma_PP_end(:));
end