
clear;clc;close all;
load("data\noisy_inversion_K_theta.mat");
%%
% For each noise level in transimission noise, plot the statistics for the
% resulting output noise via box plots

% K_error: 6x20 cell array, 6 noise levels, 20 data points per noise
% level

%===============================================================
% TO Glen: change all K_error to theta_error to plot theta error
%===============================================================



% transmission_noise_level 1x6 cell array, each entry stores a numerical noise
% level

% ---- Convert K_error cells to a numeric matrix [nLevels x nRuns] ----
[nLevels, nRuns] = size(K_error);
vals = nan(nLevels, nRuns);

for i = 1:nLevels
    for j = 1:nRuns
        v = K_error{i,j};
        if isscalar(v)
            vals(i,j) = v;
        else
            % Choose a scalar summary if v is a vector/array.
            % Options: mean(abs(v(:))), rms(v(:)), norm(v(:)), etc.
            vals(i,j) = mean(abs(v(:)));
        end
    end
end

% ---- Prepare data & groups for a single boxplot call ----
data = vals(:);                                 % (nLevels*nRuns) x 1
groupIdx = kron((1:nLevels)', ones(nRuns,1));   % same length as data

% Build x-axis labels from transmission_noise_level values
if iscell(transmission_noise_level)
    te = cellfun(@(x) x, transmission_noise_level);
else
    te = transmission_noise_level;
end
xlabels = arrayfun(@(x) sprintf('%g', x), te(:), 'UniformOutput', false);


%%
% ---- Min–Mean–Max plot with UNIFORM color norm + explicit labels ----
minv  = nanmin(vals, [], 2);
maxv  = nanmax(vals, [], 2);
meanv = nanmean(vals, 2);
N     = sum(~isnan(vals),2);

vmin = nanmin(vals(:));
vmax = nanmax(vals(:));

figure('Color','w'); hold on;
colormap(parula(256));
caxis([vmin vmax]);
cb = colorbar; ylabel(cb,'Constraint error');

cm = colormap;
idx_from_val = @(v) max(1, min(size(cm,1), round( 1 + (v - vmin) * (size(cm,1)-1) / max(eps, (vmax - vmin)) )));
val2color    = @(v) cm(idx_from_val(v), :);

cap   = 0.25;        % half-width of horizontal caps
xoff  = 0.32;        % small right offset for text labels

% legend proxies (styled; color is neutral just for legend)
phMean = plot(nan,nan,'-o','Color',[0 0 0],'LineWidth',3,'MarkerSize',5,'DisplayName','mean'); hold on;
phMin  = plot(nan,nan,'--','Color',[0.2 0.2 0.2],'LineWidth',2,'DisplayName','min');
phMax  = plot(nan,nan,':','Color',[0.2 0.2 0.2],'LineWidth',2,'DisplayName','max');

for i = 1:numel(meanv)
    x = i;

    % vertical whisker (neutral gray)
    plot([x x], [minv(i) maxv(i)], 'Color',[0.3 0.3 0.3], 'LineWidth', 2);

    % caps with distinct styles + value-mapped colors
    % min (dashed)
    plot([x-cap x+cap], [minv(i) minv(i)], '--', 'LineWidth', 2, 'Color', val2color(minv(i)));
    % max (dotted)
    plot([x-cap x+cap], [maxv(i) maxv(i)], ':',  'LineWidth', 2, 'Color', val2color(maxv(i)));
    % mean (solid + marker, thicker)
    plot([x-cap x+cap], [meanv(i) meanv(i)], '-o', 'LineWidth', 3, 'MarkerSize', 4, ...
         'Color', val2color(meanv(i)), 'MarkerFaceColor', val2color(meanv(i)));

    % text labels at right edge
    % text(x+cap+xoff, maxv(i),  'max',  'HorizontalAlignment','left','VerticalAlignment','middle');
    % text(x+cap+xoff, meanv(i), 'mean', 'HorizontalAlignment','left','VerticalAlignment','middle');
    % text(x+cap+xoff, minv(i),  'min',  'HorizontalAlignment','left','VerticalAlignment','middle');

    % (optional) n label
    % text(x, maxv(i), sprintf('  n=%d', N(i)), 'VerticalAlignment','bottom');
end

% jittered points colored by value (same uniform norm)
% data = vals(:);
% groupIdx = kron((1:numel(meanv))', ones(size(vals,2),1));
% g = groupIdx(:);
% xj = g + 0.12*(rand(size(g))-0.5);
% scatter(xj, data, 18, data, 'filled', 'MarkerFaceAlpha', 0.75);

xlim([0.5, numel(meanv)+0.5]);
xticks(1:numel(meanv)); xticklabels(xlabels);
xlabel('Transmission noise level',Interpreter='latex');
ylabel('Constraint error (per run)',Interpreter='latex');
title('Constraint error vs. transmission noise',Interpreter='latex');
grid on; box on;

legend([phMean phMin phMax],'Location','best',Interpreter='latex');  % clearly maps styles to mean/min/max
hold off;

