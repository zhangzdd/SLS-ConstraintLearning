% Plot region defined by A x >= b and overlay a reference polygon
% === New Ax ≥ b constraints ===

% Flip for Ax <= b format (used for feasibility sampling)
% A_plot = (-1)*[   -0.1560   -0.9878;
%             0.8682   -0.4961;
%             0.2747    0.9615;
%             -0.9430    0.3328];
% b_plot = [    -4.8348;
%             -9.0545;
%             -11.8130;
%             -10.3174];
function plot_region(A,b,vertices)
    A_plot = A;
    b_plot = b;
    % A_plot = [-0.2741    0.0967];
    % b_plot = -3;
    
    % Sample grid of points
    [x_grid, y_grid] = meshgrid(linspace(-20, 20, 800), linspace(-20, 20, 800));
    x_flat = [x_grid(:), y_grid(:)];
    
    % Evaluate which points satisfy all constraints
    inside = all((A_plot * x_flat') <= b_plot, 1);
    x_in = x_flat(inside, :);
    
    % Check if region exists
    if isempty(x_in)
        error('No feasible region found from A x ≥ b');
    end
    
    % Compute convex hull of feasible region
    k = convhull(x_in(:,1), x_in(:,2));
    region_x = x_in(k,1);
    region_y = x_in(k,2);
    
    % === Overlay reference polygon ===
    % 


    vertices_closed = [vertices; vertices(1,:)];
    
    % === Plot ===
    figure; hold on; axis equal;
    
    % Region from Ax ≥ b
    fill(region_x, region_y, [0.2, 0.6, 0.8], ...
        'FaceAlpha', 0.4, 'EdgeColor', 'k', 'LineWidth', 2);
    
    % Overlay polygon
    plot(vertices_closed(:,1), vertices_closed(:,2), 'r--', 'LineWidth', 2);
    scatter(vertices(:,1), vertices(:,2), 60, 'ro', 'filled');
    
    % Labels and aesthetics
    xlabel('x'); ylabel('y');
    title('Safe Region');
    % Ax >= b
    % legend('Learned unsafe Region', 'Ground Truth Region', 'Vertices', 'Location', 'best');
    grid on;
end
