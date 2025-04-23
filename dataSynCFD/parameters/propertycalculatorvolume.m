T = 25;
V_total = 80;

% volume fraction wise density/viscosity calculation
v_g_all = linspace(0, 1, 10000);

% Compute viscosity list (element-wise)
eta_all = zeros(size(v_g_all));
for i = 1:length(v_g_all)
    fraction = v_g_all(i);  % pass scalar value
    [~, eta_all(i)] = propertycalculatorVOLUME(fraction, T);
end

log_eta_all = log(eta_all);
% log_eta_all = eta_all;

%% Plotting
figure;

subplot(1, 2, 1);
plot(v_g_all, log_eta_all, 'b', 'LineWidth', 2);
xlabel('Glycerol Volume Fraction');
ylabel('log(Viscosity) [ln(Ns/m^2)]');
title('log(Viscosity) vs. Glycerol Volume Fraction');
grid on;

subplot(1, 2, 2);
plot(v_g_all, eta_all, 'r', 'LineWidth', 2);
xlabel('Glycerol Volume Fraction');
ylabel('Viscosity [Ns/m^2]');
title('Viscosity vs. Glycerol Volume Fraction');
grid on;

% --- Interpolation at evenly spaced log-viscosity points ---
log_eta_targets = linspace(min(log_eta_all), max(log_eta_all), 13);
v_g_interp = interp1(log_eta_all, v_g_all, log_eta_targets, 'linear');

% Output
disp('Interpolated glycerol volume fractions at evenly spaced log(viscosity):');
disp(v_g_interp);

% Calculate weight fractions for only the 10 interpolated volume fractions
rho_g = 1273.3 - 0.6121 * T;
rho_w = 1000 * (1 - ((abs(T - 3.98)) / 615)^1.71);
w_g_interp = (rho_g .* v_g_interp) ./ (rho_g .* v_g_interp + rho_w .* (1 - v_g_interp));
rho_interp = zeros(size(v_g_interp));
eta_interp = zeros(size(v_g_interp));

% Calculate density and viscosity for each interpolated volume fraction
for i = 1:length(v_g_interp)
    [rho_interp(i), eta_interp(i)] = propertycalculatorVOLUME(v_g_interp(i), T);
end

[m_glyc, m_water] = calculate_masses_from_weight_fraction(w_g_interp, V_total, T);

eta_cP = eta_interp * 1000;
[m_glyc, m_water] = calculate_masses_from_weight_fraction(w_g_interp, V_total, T);
surfT = 0.0762 * ones(size(w_g_interp));  % Surface tension in N/m

% Create table
result_table = table(w_g_interp', m_glyc', m_water', rho_interp', eta_interp', eta_cP', surfT', ...
    'VariableNames', {'WeightFraction', 'Glycerol_mass_g', 'Water_mass_g', ...
                      'Density_kg_per_m3', 'Viscosity_Ns_per_m2', 'Viscosity_cP', 'SurfaceTension_N_per_m'});

% Display table
disp(result_table);

% Save as CSV
writetable(result_table, 'glycerol_water_properties.csv');
%% function definition
function[rho,eta]=propertycalculatorVOLUME(fraction_glyc,T)

volume_glycerol=fraction_glyc;
volume_water=1-fraction_glyc;

% Calculations:
total_volume=volume_glycerol+volume_water;
volume_fraction=volume_glycerol/total_volume;

%density_glycerol=1277-0.654*T;  % kg/m^3, equation 24
density_glycerol=1273.3-0.6121*T;  % UPDATED following Andreas Volk’s suggestion

density_water=1000*(1-((abs(T-3.98))/615)^1.71);  % UPDATED following A.V.'s suggestion

mass_glycerol=density_glycerol*volume_glycerol; % kg
mass_water=density_water*volume_water; % kg
total_mass=mass_glycerol+mass_water; % kg
mass_fraction=mass_glycerol/total_mass;

viscosity_glycerol=0.001*12100*exp((-1233+T)*T/(9900+70*T)); % equation 22. Note factor of 0.001 -> converts to Ns/m^2
viscosity_water=0.001*1.790*exp((-1230-T)*T/(36100+360*T)); % equation 21. Again, note conversion to Ns/m^2

a=0.705-0.0017*T;
b=(4.9+0.036*T)*a^2.5;
alpha=1-mass_fraction+(a*b*mass_fraction*(1-mass_fraction))/(a*mass_fraction+b*(1-mass_fraction));
A=log(viscosity_water/viscosity_glycerol); % Note this is NATURAL LOG (ln), not base 10.
viscosity_mix=viscosity_glycerol*exp(A*alpha); % Ns/m^2, equation 6

% Andreas Volk polynomial:
c=1.78E-6*T.^2-1.82E-4*T+1.41E-2;
contraction=1+(c.*sin((mass_fraction).^1.31.*pi).^0.81);

density_mix=(density_glycerol*fraction_glyc+density_water*(1-fraction_glyc))*contraction; % equation 25

eta=viscosity_mix;
rho=density_mix;

end

function [mass_glycerol, mass_water] = calculate_masses_from_weight_fraction(w_g_all, V_total, T)
% Calculate the mass of glycerol and water to mix for given weight fractions
% Inputs:
%   w_g_all  - vector of weight fractions (e.g. from 0 to 1)
%   V_total  - total solution volume (same for each sample, e.g., 100 mL or 0.1 L)
%   T        - temperature in °C for accurate density calculation
% Outputs:
%   mass_glycerol - vector of glycerol masses [g or kg]
%   mass_water    - vector of water masses [g or kg]

% Densities (kg/m^3) at temperature T
rho_g = 1273.3 - 0.6121 * T;
rho_w = 1000 * (1 - ((abs(T - 3.98)) / 615)^1.71);

% Convert to g/mL (since volume may be in mL, and we want grams)
rho_g = rho_g / 1000;  % g/mL
rho_w = rho_w / 1000;  % g/mL

% Total mass of solution for each weight fraction
% w_g = m_g / (m_g + m_w)
% => m_g = w_g * m_total
% => m_w = (1 - w_g) * m_total

% Assume final volume V_total = m_total / density_mix
% So m_total = V_total * density_mix

density_mix = @(w) (rho_g .* w + rho_w .* (1 - w));  % approx mix density

m_total = V_total .* density_mix(w_g_all);          % in grams
mass_glycerol = w_g_all .* m_total;
mass_water = (1 - w_g_all) .* m_total;

end