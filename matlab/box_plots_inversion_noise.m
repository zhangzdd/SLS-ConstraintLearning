load("data\transmission_noise_1\noisy_inversion_6_20.mat");

% For each noise level in transimission noise, plot the statistics for the
% resulting output noise via box plots

% theta_error: 6x20 cell array, 6 noise levels, 20 data points per noise
% level

% transmission_noise_level 1x6 cell array, each entry stores a numerical noise
% level

% ---- Convert theta_error cells to a numeric matrix [nLevels x nRuns] ----
[nLevels, nRuns] = size(theta_error);
vals = nan(nLevels, nRuns);

for i = 1:nLevels
    for j = 1:nRuns
        v = theta_error{i,j};
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

% ---- Plot ----
figure('Color','w'); 
boxplot(data, groupIdx, 'Labels', xlabels, 'Whisker', 1.5);
xlabel('Transmission noise level');
ylabel('Constraint error (summary per run)');
title('Constraint error vs. transmission noise (box plots)');

% Optional: overlay individual points with jitter for visibility
hold on;
g = groupIdx(:);
xj = g + 0.15*(rand(size(g))-0.5);    % small horizontal jitter
plot(xj, data, '.', 'MarkerSize', 10);
hold off;

% Optional: show medians numerically in the console
med_per_level = median(vals, 2);
disp(table(te(:), med_per_level, 'VariableNames', {'TransmissionNoise','MedianError'}));

%%
% ---- Rich per-level summary table ----
stats = table();
nLevels = numel(te);

% Preallocate
N          = zeros(nLevels,1);
NMissing   = zeros(nLevels,1);
MeanErr    = nan(nLevels,1);
MedianErr  = nan(nLevels,1);
StdErr     = nan(nLevels,1);
MADErr     = nan(nLevels,1);         % median absolute deviation
Q1         = nan(nLevels,1);
Q3         = nan(nLevels,1);
IQRv       = nan(nLevels,1);
MinErr     = nan(nLevels,1);
MaxErr     = nan(nLevels,1);
P10        = nan(nLevels,1);
P90        = nan(nLevels,1);
NOutliers  = nan(nLevels,1);
SEM        = nan(nLevels,1);         % standard error of the mean
CI95_L     = nan(nLevels,1);
CI95_U     = nan(nLevels,1);

for i = 1:nLevels
    di_all = vals(i,:);                 % all runs for this level (may include NaN)
    NMissing(i) = sum(isnan(di_all));
    di = di_all(~isnan(di_all));        % valid data only
    N(i) = numel(di);
    if N(i) > 0
        MeanErr(i)   = mean(di);
        MedianErr(i) = median(di);
        StdErr(i)    = std(di, 0);      % sample std
        MADErr(i)    = mad(di, 1);      % MAD with scaling=1
        MinErr(i)    = min(di);
        MaxErr(i)    = max(di);
        P10(i)       = prctile(di,10);
        P90(i)       = prctile(di,90);
        q = prctile(di,[25 75]);
        Q1(i)  = q(1);
        Q3(i)  = q(2);
        IQRv(i)= Q3(i)-Q1(i);

        % Boxplot outliers (Tukey 1.5*IQR)
        lf = Q1(i) - 1.5*IQRv(i);
        uf = Q3(i) + 1.5*IQRv(i);
        NOutliers(i) = sum(di < lf | di > uf);

        % 95% CI for the mean (normal approx)
        if N(i) > 1
            SEM(i)   = StdErr(i)/sqrt(N(i));
            hw       = 1.96*SEM(i);
            CI95_L(i)= MeanErr(i) - hw;
            CI95_U(i)= MeanErr(i) + hw;
        end
    end
end

stats = table( ...
    te(:), N, NMissing, MeanErr, MedianErr, StdErr, MADErr, ...
    Q1, Q3, IQRv, MinErr, MaxErr, P10, P90, NOutliers, SEM, CI95_L, CI95_U, ...
    'VariableNames', {'TransmissionNoise','N','NMissing','Mean','Median','Std','MAD', ...
                      'Q1','Q3','IQR','Min','Max','P10','P90','NOutliers','SEM','CI95_L','CI95_U'});

% Sort by noise level (optional)
stats = sortrows(stats, 'TransmissionNoise');

disp(stats);

% Optional: write to CSV for sharing
% writetable(stats, fullfile(pwd,'boxplot_stats.csv'));

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
xlabel('Transmission noise level');
ylabel('Constraint error (per run)');
title('Constraint error vs. transmission noise (min–mean–max, uniform color norm)');
grid on; box on;

legend([phMean phMin phMax],'Location','best');  % clearly maps styles to mean/min/max
hold off;

