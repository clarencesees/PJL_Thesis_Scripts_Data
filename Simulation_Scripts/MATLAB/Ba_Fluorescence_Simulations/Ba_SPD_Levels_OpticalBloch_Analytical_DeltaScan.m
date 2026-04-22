Delta_S_array = -100:1:0; %A range of frequency detunings for the S-P transition laser in MHz.
Delta_D_array = -100:1:100; %A range of frequency detunings for the D-P transition laser in MHz.
sigma_PP_end = nan(numel(Delta_S_array),numel(Delta_D_array)); %Initialize array for the P state populations.
for hh=1:numel(Delta_S_array)
    for hhh=1:numel(Delta_D_array)
        Delta_S = 2*pi*Delta_S_array(hh); %Scale frequency by 2*pi
        Delta_D = 2*pi*Delta_D_array(hhh); %Scale frequency by 2*pi
        Ba_SPD_Levels_OpticalBloch_Analytical; %Compute steady-state P state population.
        sigma_PP_end(hh,hhh) = sigma_PP; %Plug computed value into array
    end
end
sigma_PP_end(isnan(sigma_PP_end)) = 0; %Analytical form may have values divided by zero during the computation, giving NaN. Replace them with zero, which is inspected to be the asymptotic value