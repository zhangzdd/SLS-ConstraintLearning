clc;clear;close all;

T = 20; % Time horizon
Ts = 0.1; % Sampling time
state_dim = 4;
input_dim = 2;
disturbance_level = 1; % The magnitude of disturbances

% Double integrator dynamics
A_t = [0 0 1 0;0 0 0 1; 0 0 0 0; 0 0 0 0]*Ts+ eye(state_dim);
% B_t = [1/2*Ts^2 0;0 1/2*Ts^2;Ts 0;0 Ts]; 
% Omit the second order terms
B_t = [0 0;0 0;1 0;0 1]*Ts;


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
A_poly = [1 0 0 0;
         -1 0 0 0;
         0 1 0 0;
         0 -1 0 0];
b_poly = [5;
        5;
        2;  
        2];
DimAffine = 1;

% Nominal trajectory z,v
z = sdpvar(state_dim*T,1);
v = sdpvar(input_dim*T,1);

constrained_states_upper_limited = 4;
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
constraints = [constraints, z_init(1:state_dim) == [0;0;0;0]];
constraints = [constraints, z_terminal(1:state_dim) == [0.;10;0;0]];
% constraints = [constraints, v_terminal(1:input_dim) == [0;0]];


% Terminal robust constraint
for dim = 1:DimAffine
    robust_affine_constraints = 0;
    for j = 1:T
        robust_affine_constraints = robust_affine_constraints + disturbance_level * norm(A_poly(dim,:)*phi_x((T-1)*state_dim+1:T*state_dim,(j-1)*state_dim+1:j*state_dim),1);
    end
    constraints = [constraints, A_poly(dim,:)*z_terminal + robust_affine_constraints <= b_poly(dim)];
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
    objective = objective + sum((z_k_plus_1(1:2) - z_k(1:2)).^2) - z_k(1);
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
xlim([-10,10]);ylim([-10,10]);
%% Run roll outs at multiple times with noise signal
num_rollout = 100;
error_signal_feedback = cell(num_rollout,1);
error_signal_openloop = cell(num_rollout,1);
state_trajectory_closedloop = cell(num_rollout,1);
input_trajectory_closedloop = cell(num_rollout,1);

for rollout_cnt = 1:num_rollout
    % Roll out trajectory with noise to the dynamics, WITHOUT the feedback
    x_init = [0;0;0;0]; % same as nominal trajectory initial state
    x = zeros(state_dim,T);
    
    for i = 1:T-1
        % noise = randn(4, 1);     % random vector from N(0,1)
        noise = disturbance_level * (rand(4,1)*2 - 1);
        new_x = A_t*x(:,i) + B_t*unstacked_v(:,i) + noise;
        x(:,i+1) = new_x;
    end
    plot(x(1,:),x(2,:),"r","DisplayName","Disturbed");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"r","DisplayName","Error signal without Feedback Control")
    error_signal_openloop{rollout_cnt} = [x(1,1:end) - unstacked_z(1,1:end);x(2,1:end) - unstacked_z(2,1:end)];
    
    % Roll out trajectory with noise to the dynamics, WITH the feedback
    
    % Suppress the NaNs
    val_phi_u = value(phi_u);
    val_phi_u(isnan(val_phi_u)) = 0;
    val_phi_x = value(phi_x);
    
    
    K = val_phi_u / val_phi_x;
    
    x_init = disturbance_level * (rand(4,1)*2 - 1);
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
        noise = disturbance_level * (rand(4,1)*2 - 1);
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
xline(b_poly(1),":",'LineWidth',2);
save('forward_game_data.mat','state_trajectory_closedloop','input_trajectory_closedloop','unstacked_z','unstacked_v','state_dim','input_dim','DimAffine','T','num_rollout','val_phi_u','val_phi_x','Z','A','B','A_t','B_t', ...
    'disturbance_level');
xlabel('x',Interpreter='latex');
ylabel('y',Interpreter='latex');

hNom = plot(unstacked_z(1,1:end),unstacked_z(2,1:end), "g", "DisplayName","Nominal Trajectory",'LineWidth',3);
lineLearn = xline(b_poly(1),"--",'LineWidth',2,'Color','k','DisplayName','Learned Constraint(s)');
lineTruth = xline(b_poly(1),'LineWidth',2,'Color','y','DisplayName','Ground Truth Constraint(s)');
hOpen = plot(nan, nan, 'r', 'DisplayName','Open-loop (disturbed)');
hFB   = plot(nan, nan, 'b', 'DisplayName','Feedback (disturbed)');
hStart = plot(unstacked_z(1,1), unstacked_z(2,1), 'o', 'LineStyle','none', ...
    'MarkerSize',8, 'MarkerFaceColor','g', 'MarkerEdgeColor','k', ...
    'DisplayName','Start');

hGoal  = plot(unstacked_z(1,end), unstacked_z(2,end), 'p', 'LineStyle','none', ...
    'MarkerSize',11, 'MarkerFaceColor','g', 'MarkerEdgeColor','k', ...
    'DisplayName','Goal');

legend([hNom hOpen hFB lineLearn lineTruth], 'Interpreter','latex', 'Location','best');



%% 
% 1) Axes cosmetics & LaTeX
ax = gca; box on; grid on; ax.Layer = 'top';
ax.TickDir = 'out'; ax.LineWidth = 1;
ax.FontName = 'Times New Roman'; ax.FontSize = 10;  % or your journal’s font
set(ax,'TickLabelInterpreter','latex');
axis equal;xlim([-10,20]);ylim([-2,20]);


% 2) Shade the forbidden half-space (to the right of the constraint line)
yl = ylim; xr = xlim;
hForbid = patch([b_poly(1) xr(2) xr(2) b_poly(1)], [yl(1) yl(1) yl(2) yl(2)], ...
    [0 0 0], 'FaceAlpha',0.06, 'EdgeColor','none', 'HandleVisibility','off');
uistack(hForbid,'bottom');  % keep it behind the trajectories
text(b_poly(1)+4, mean(yl), '\textbf{unsafe}', 'Interpreter','latex', ...
     'Rotation',45, 'HorizontalAlignment','left', 'Color',[0 0 0],'FontSize',30);
% %%
% % Compute the maximum error tube
% constraints = []; % Empty the constraints to perform a new optimization
% val_phi_x = value(phi_x);
% w = sdpvar(state_dim*T,1);
% x_tube = zeros(1,T);
% y_tube = zeros(1,T);
% for t = 1:T
%     x_bound = 0;
%     y_bound = 0;
%     for j = 1:t
%         x_bound = x_bound + norm([1 0 0 0] * val_phi_x((t-1)*state_dim+1:t*state_dim,(j-1)*state_dim+1:j*state_dim) * w((j-1)*state_dim+1:j*state_dim),1);
%         y_bound = y_bound + norm([0 1 0 0] * val_phi_x((t-1)*state_dim+1:t*state_dim,(j-1)*state_dim+1:j*state_dim) * w((j-1)*state_dim+1:j*state_dim),1);
%         constraints = [constraints, norm(w((j-1)*state_dim+1:j*state_dim),1) <= 1];
%     end
%     % Try to maximize the bounds
%     optimize(constraints, 10000 - norm(x_bound,1), ops);
%     x_tube(t) = value(x_bound);
%     optimize(constraints, 10000 - norm(y_bound,1), ops);
%     y_tube(t) = value(y_bound);
% end
% %% Visualize the tube bound (this section could take a while to run)
% % Plot the tube along with the various rollout trajectories
% figure(2);hold on;
% for t = 1:(T-1)
%     % Upper bound line from x_tube(t) to x_tube(t+1)
%     plot([t, t+1], [x_tube(t), x_tube(t+1)], 'k', 'LineWidth', 2);
% 
%     % Lower bound line from -x_tube(t) to -x_tube(t+1)
%     plot([t, t+1], [-x_tube(t), -x_tube(t+1)], 'k', 'LineWidth', 2);
% end
% 
% % Plot rollout trajectories
% for rollout_cnt = 1:num_rollout
%     for t = 1:T
%         x_closed_loop = error_signal_feedback{rollout_cnt}(1,t);
%         x_open_loop   = error_signal_openloop{rollout_cnt}(2,t);
%         scatter(t, x_closed_loop, 10, 'blue');
%         scatter(t, x_open_loop,   10, 'red');
% 
%     end
% end
% xlabel("t");ylabel("x");