clc;clear;close all;

T = 10; % Time horizon
Ts = 0.1; % Sampling time
state_dim = 12;
input_dim = 4;
disturbance_level = 1; % The magnitude of disturbances

% Parameters
g  = 9.81;    % gravity [m/s^2]
m  = 1.0;     % mass [kg]
Ix = 0.02;    % inertia about x [kg*m^2]
Iy = 0.02;    % inertia about y [kg*m^2]
Iz = 0.04;    % inertia about z [kg*m^2]

% State: [x y z phi theta psi vx vy vz p q r]'
% Input: [tau_x tau_y tau_z T]'

A_t = zeros(12,12);
A_t(1,7)  = 1;        % xdot = vx
A_t(2,8)  = 1;        % ydot = vy
A_t(3,9)  = 1;        % zdot = vz
A_t(4,10) = 1;        % phidot = p
A_t(5,11) = 1;        % thetadot = q
A_t(6,12) = 1;        % psidot = r
A_t(7,5)  =  g;       % vxdot ≈  g*theta
A_t(8,4)  = -g;       % vydot ≈ -g*phi
% remaining entries are zero

B_t = zeros(12,4);
B_t(10,1) = 1/Ix;     % pdot = tau_x/Ix
B_t(11,2) = 1/Iy;     % qdot = tau_y/Iy
B_t(12,3) = 1/Iz;     % rdot = tau_z/Iz
B_t(9,4)  = 1/m;      % vzdot = T/m - g  (the -g is in A via equilibrium)


A_t = eye(12) + Ts*A_t;
B_t = Ts*B_t;


num_repetitions_A = T - 1;
num_repetitions_B = T - 1;
matrices_A = cell(1, num_repetitions_A);
for i = 1:num_repetitions_A+1
    if i == T
        matrices_A{i} = zeros(size(A_t));
    else
        matrices_A{i} = A_t;
    end
end
matrices_B = cell(1, num_repetitions_B);
for i = 1:num_repetitions_B+1
    if i == T
        matrices_B{i} = zeros(size(B_t));
    else
        matrices_B{i} = B_t;
    end
end
A = blkdiag(matrices_A{:});
B = blkdiag(matrices_B{:});

Z = createBlockDownshiftOperator(state_dim,T);

% Construct the phi block matrices
phi_x = sdpvar(T*state_dim, T*state_dim,'full');
phi_u = sdpvar(T*input_dim, T*state_dim,'full');

constraints = [];

% Populate the phi_x and phi_u as blcok lower triangular matrices
for i = 1:T
    for j = 1:T
        if j>i
            constraints = [constraints,  phi_x((i-1)*state_dim+1:i*state_dim,(j-1)*state_dim+1:j*state_dim) == zeros(state_dim,state_dim)];
            constraints = [constraints,  phi_u((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) == zeros(input_dim,state_dim)];
            % phi_x((i-1)*state_dim+1:i*state_dim,(j-1)*state_dim+1:j*state_dim) = zeros(state_dim,state_dim);
            % phi_u((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) = zeros(input_dim,state_dim);
        end
    end
end

% % "Tube" region specified with vertices
% vertices = [
%     0,  0;
%     7,  3;
%     10,   10;
%     3,  7
% ];
% 
% % The number of affine constraints
% DimAffine = size(vertices,1);
% 
% % Converted to halfspace form: A x <= b
% n_edges = size(vertices, 1);
% A_poly = zeros(n_edges, 2);
% b_poly = zeros(n_edges, 1);
% for i = 1:n_edges
%     p1 = vertices(i,:);
%     p2 = vertices(mod(i,n_edges)+1,:);
%     edge = p2 - p1;
%     normal = [edge(2), -edge(1)];  % outward normal (CW)
%     normal = normal / norm(normal);
%     A_poly(i,:) = normal;
%     b_poly(i) = normal * p1';
% end
% plot_region(A_poly,b_poly,vertices);

% Tube affine constraint in the MPSF paper
A_poly = [1 0 0 0 0 0 0 0 0 0 0 0 ;
         0 1 0 0 0 0 0 0 0 0 0 0;
         0 0 1 0 0 0 0 0 0 0 0 0];
b_poly = [7;
        7;
        55];
DimAffine = 3;

% Nominal trajectory z,v
z = sdpvar(state_dim*T,1);
v = sdpvar(input_dim*T,1);

constrained_states_upper_limited = 4;
tube_size = sdpvar(DimAffine,T);
for dim = 1:DimAffine
    for k = 1:T-1
        phi_kx = phi_x((k-1)*state_dim+1:k*state_dim,:);
        phi_ku = phi_u((k-1)*input_dim+1:k*input_dim,:);
        z_k = z((k-1)*state_dim+1:k*state_dim);
        v_k = v((k-1)*input_dim+1:k*input_dim);
        z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
        constraints = [constraints, z_k_plus_1 == A_t*z_k + B_t*v_k];
        % constraints = [constraints, A_poly(dim,:)*z_k(1:constrained_states_upper_limited) + norm(A_poly(dim,:)*phi_kx(1:constrained_states_upper_limited,:),1) <= b_poly(dim)];
        robust_affine_constraints = 0;
        for j = 1:k
            robust_affine_constraints = robust_affine_constraints + disturbance_level * norm(A_poly(dim,:)*phi_kx(:,(j-1)*state_dim+1:j*state_dim),1);
        end
        constraints = [constraints, A_poly(dim,:)*z_k + robust_affine_constraints <= b_poly(dim)];
        tube_size(dim, k) = robust_affine_constraints;
        % constraints = [constraints, [1 0]*v_k + norm([1 0]*phi_ku,1) <= 75];
        % constraints = [constraints, [1 0]*v_k - norm([1 0]*phi_ku,1) >= -75];
        % constraints = [constraints, [0 1]*v_k + norm([0 1]*phi_ku,1) <= 75];
        % constraints = [constraints, [0 1]*v_k - norm([0 1]*phi_ku,1) >= -75];
    end
end

% Start and terminal constraint
z_init = z((1-1)*state_dim+1:1*state_dim);
z_terminal = z((T-1)*state_dim+1:T*state_dim);
% v_terminal = v((T-1)*input_dim+1:T*input_dim);
constraints = [constraints, z_init(1:state_dim) == zeros(12,1)];
constraints = [constraints, z_terminal(1:3) == [0; 0; 50]];
% constraints = [constraints, v_terminal(1:input_dim) == [0;0]];


% Terminal robust constraint
for dim = 1:DimAffine
    robust_affine_constraints = 0;
    for j = 1:T
        robust_affine_constraints = robust_affine_constraints + disturbance_level * norm(A_poly(dim,:)*phi_x((T-1)*state_dim+1:T*state_dim,(j-1)*state_dim+1:j*state_dim),1);
    end
    constraints = [constraints, A_poly(dim,:)*z_terminal + robust_affine_constraints <= b_poly(dim)];
    tube_size(dim,T) = robust_affine_constraints;
end


% SLS constraints
constraints = [constraints, [eye(T*state_dim) - Z*A, -Z*B]*[phi_x;phi_u] == eye(T*state_dim)];


% Objective
objective = 0;
for k = 1:T-1
    z_k = z((k-1)*state_dim+1:k*state_dim);
    z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
    
    % The objective is two-fold: trying to maximize x-coordinates, and
    % trying to smooth the overall trajectory
    objective = objective + sum((z_k_plus_1(1:3) - z_k(1:3)).^2) - z_k(1) - z_k(2);
end

% Solve
ops = sdpsettings('solver','gurobi','verbose', 2);
optimization_results = optimize(constraints, objective, ops);
%%
% Plot nominal trajectory (z)
figure;hold on;
unstacked_z = reshape(value(z),[state_dim, T]);
unstacked_v = reshape(value(v),[input_dim, T]);
% plot(unstacked_z(1,:),unstacked_z(2,:),"g","DisplayName","Nominal");
% plot(unstacked_z(1,:) + value(tube_size),unstacked_z(2,:),'Color','m','DisplayName','Tube Bound',LineWidth=3)

xlim([-10,10]);ylim([-10,10]);
%% Run roll outs at multiple times with noise signal
num_rollout = 100;
error_signal_feedback = cell(num_rollout,1);
error_signal_openloop = cell(num_rollout,1);
state_trajectory_closedloop = cell(num_rollout,1);
input_trajectory_closedloop = cell(num_rollout,1);

for rollout_cnt = 1:num_rollout
    % Roll out trajectory with noise to the dynamics, WITHOUT the feedback
    x_init = zeros(12,1); % same as nominal trajectory initial state
    x = zeros(state_dim,T);
    
    for i = 1:T-1
        % noise = randn(4, 1);     % random vector from N(0,1)
        noise = disturbance_level * (rand(state_dim,1)*2 - 1);
        new_x = A_t*x(:,i) + B_t*unstacked_v(:,i) + noise;
        x(:,i+1) = new_x;
    end
    % plot(x(1,:),x(2,:),"r","DisplayName","Disturbed");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"r","DisplayName","Error signal without Feedback Control")
    error_signal_openloop{rollout_cnt} = [x(1,1:end) - unstacked_z(1,1:end);x(2,1:end) - unstacked_z(2,1:end)];
    
    % Roll out trajectory with noise to the dynamics, WITH the feedback
    
    % Suppress the NaNs
    val_phi_u = value(phi_u);
    val_phi_u(isnan(val_phi_u)) = 0;
    val_phi_x = value(phi_x);
    
    
    K = val_phi_u / val_phi_x;
    
    x_init = disturbance_level * (rand(state_dim,1)*2 - 1);
    x = zeros(state_dim,T);
    x(:,1) = x_init;
    feedback_u = zeros(input_dim,T);
    
    for i = 1:T
        feedback_control = zeros(input_dim,1);
        for j = 1:i
            % Feedback gain exists for all previous timesteps (causal system)
            feedback_control = feedback_control + K((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim)*(x(:,j) - unstacked_z(:,j));
        end
        % noise = randn(4, 1);     % random vector from N(0,1)
        % noise = noise / norm(noise,1);     % normalize to have norm 1
        noise = disturbance_level * (rand(state_dim,1)*2 - 1);
        if i<=T-1
            x(:,i+1) = A_t*x(:,i) + B_t*(feedback_control + unstacked_v(:,i)) + noise;
        end
        feedback_u(:,i) = feedback_control;
    end
    plot(x(1,1:end),x(2,1:end),"b","DisplayName","Disturbed w/ Feedback Control");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"b","DisplayName","Error signal w/ Feedback Control")
    error_signal_feedback{rollout_cnt} = [x(1,1:end) - unstacked_z(1,1:end);x(2,1:end) - unstacked_z(2,1:end)];
    state_trajectory_closedloop{rollout_cnt} = x;
    input_trajectory_closedloop{rollout_cnt} = feedback_u;
end


%% 3D visualization (replaces the old 2D plotting below the TODO)

% New 3D figure (separate from any earlier figure state)
figure; hold on;

% Nominal, start, goal
hNom = plot3(unstacked_z(1,:), unstacked_z(2,:), unstacked_z(3,:), ...
    "g", "DisplayName","Nominal Trajectory", 'LineWidth',3);

hStart = plot3(unstacked_z(1,1), unstacked_z(2,1), unstacked_z(3,1), ...
    'o', 'LineStyle','none', 'MarkerSize',8, ...
    'MarkerFaceColor','g', 'MarkerEdgeColor','k', 'DisplayName','Start');

hGoal  = plot3(unstacked_z(1,end), unstacked_z(2,end), unstacked_z(3,end), ...
    'p', 'LineStyle','none', 'MarkerSize',11, ...
    'MarkerFaceColor','g', 'MarkerEdgeColor','k', 'DisplayName','Goal');

% Plot all closed-loop disturbed trajectories (from the stored cell array)
for rr = 1:num_rollout
    xrr = state_trajectory_closedloop{rr};
    if ~isempty(xrr)
        scatter3(xrr(1,:), xrr(2,:), xrr(3,:), "b", "HandleVisibility","off");
    end
end
% Legend placeholder for feedback trajectories
hFB = plot3(nan, nan, nan, 'b', 'DisplayName','Feedback (disturbed)');

% Axes cosmetics
grid on; box on; ax = gca; ax.Layer = 'top';
ax.TickDir = 'out'; ax.LineWidth = 1;
ax.FontName = 'Times New Roman'; ax.FontSize = 10;
set(ax,'TickLabelInterpreter','latex');

xlabel('x','Interpreter','latex');
ylabel('y','Interpreter','latex');
zlabel('z','Interpreter','latex');
% axis vis3d; 
axis equal;
xlim([-10, 20]); ylim([-10, 20]); zlim([0, 55]);
view(3);

% --- Constraint planes from A_poly * [x y z ...]^T <= b_poly ---
% Ensure z range includes z = b_poly(3)
zl = zlim; 
zlim([zl(1), max(zl(2), b_poly(3))]);

% Refresh limits after any change
xl = xlim; yl = ylim; zl = zlim;

% x <= 7  -> plane x = 7
XX = [b_poly(1) b_poly(1) b_poly(1) b_poly(1)];
YY = [yl(1)     yl(2)     yl(2)     yl(1)];
ZZ = [zl(1)     zl(1)     zl(2)     zl(2)];
lineLearn = patch('XData',XX,'YData',YY,'ZData',ZZ, ...
    'FaceColor','k','FaceAlpha',0.06,'EdgeColor','k', ...
    'DisplayName','Learned Constraint(s)');

% y <= 7  -> plane y = 7
YY = [b_poly(2) b_poly(2) b_poly(2) b_poly(2)];
XX = [xl(1)     xl(2)     xl(2)     xl(1)];
ZZ = [zl(1)     zl(1)     zl(2)     zl(2)];
patch('XData',XX,'YData',YY,'ZData',ZZ, ...
    'FaceColor','k','FaceAlpha',0.06,'EdgeColor','k', ...
    'HandleVisibility','off');

% z <= 55 -> plane z = 55
ZZ = [b_poly(3) b_poly(3) b_poly(3) b_poly(3)];
XX = [xl(1)     xl(2)     xl(2)     xl(1)];
YY = [yl(1)     yl(1)     yl(2)     yl(2)];
patch('XData',XX,'YData',YY,'ZData',ZZ, ...
    'FaceColor','k','FaceAlpha',0.06,'EdgeColor','k', ...
    'HandleVisibility','off');

% Legend (no duplicate x-plane)
legend([hNom hFB lineLearn], 'Interpreter','latex', 'Location','best');



%% Plot tubes separately (time vs one coordinate, keep as in 2D)

% TODO: now we need to plot boxes using value from tube_size




tube_vals = value(tube_size);  % [DimAffine x T], here DimAffine = 3 for x,y,z
if ~isempty(tube_vals) && size(tube_vals,1) >= 3
    % Optionally, decimate to reduce clutter (set to 1 to draw all)
    STEP = 1;  

    for k = 1:STEP:T
        c = unstacked_z(1:3, k);              % center: [x;y;z] at time k
        r = max(tube_vals(1:3, k), 0);        % half-sizes (ensure nonnegative)

        if any(r > 0)
            % Box corner ranges
            xr = [c(1)-r(1), c(1)+r(1)];
            yr = [c(2)-r(2), c(2)+r(2)];
            zr = [c(3)-r(3), c(3)+r(3)];

            % 8 vertices (using ndgrid to enumerate corners)
            [Xv, Yv, Zv] = ndgrid(xr, yr, zr);
            V = [Xv(:), Yv(:), Zv(:)];  % 8x3

            % 6 faces (each as 4 vertex indices)
            % Order matches the 8 vertices generated by ndgrid:
            % idx mapping: (x,y,z) -> 1:(-,-,-), 2:(+,-,-), 3:(-,+,-), 4:(+,+,-),
            %                                 5:(-,-,+), 6:(+,-,+), 7:(-,+,+), 8:(+,+,+)
            F = [1 3 7 5;   % -X side
                 2 4 8 6;   % +X side
                 1 2 4 3;   % -Z side (bottom)
                 5 6 8 7;   % +Z side (top)
                 1 2 6 5;   % -Y side
                 3 4 8 7];  % +Y side

            % Draw as a wireframe box (no fill) to avoid hiding trajectories
            patch('Vertices', V, 'Faces', F, ...
                  'FaceColor','none', ...
                  'EdgeColor',[0 0.4470 0.7410], ...   % MATLAB default blue
                  'LineWidth', 0.8, ...
                  'HandleVisibility','off');
        end
    end
end

