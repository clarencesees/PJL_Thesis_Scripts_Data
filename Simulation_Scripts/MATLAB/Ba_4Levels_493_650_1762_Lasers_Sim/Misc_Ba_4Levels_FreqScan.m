Delta_SD2_array = -1000:1:1000;
sigma_end_array = nan(size(Delta_SD2_array));

for Delta_SD2_count = 1:numel(Delta_SD2_array)
    Delta_SD2 = 2*pi*Delta_SD2_array(Delta_SD2_count);
    Ba_4Levels_Shelving_Sim_linsolve;
    sigma_end_array(Delta_SD2_count) = sigma_end(cp1.*cp1_conj == 1);
end