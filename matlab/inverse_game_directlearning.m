% Load the forward game data
clf;clear;
load("forward_game_data.mat");
num_rollout = 20;

%% Write the KKT conditions for robust-constraints-learning

constraints = [];
% Nominal trajectory z,v



% Populate the t_all vector, which is used to compute jacobians
t_all = [];
t_all_numerical = [];
t_demo = cell(num_rollout,1);
z = cell(num_rollout,1);
v = cell(num_rollout,1);
for rollout_cnt = 1:num_rollout
    z{rollout_cnt} = sdpvar(state_dim*T,1);
    v{rollout_cnt} = sdpvar(input_dim*T,1);
    t_demo{rollout_cnt} = [z{rollout_cnt};v{rollout_cnt}];
    t_all = [t_all;t_demo{rollout_cnt}];
    t_all_numerical = [t_all_numerical;[state_trajectory_closedloop{rollout_cnt}(:);input_trajectory_closedloop{rollout_cnt}(:)]];
end

% A_poly = sdpvar(1,4);
A_poly = [1 0 0 0];
b_poly = sdpvar(1,1);

lambda_affine = sdpvar(T, num_rollout,'full');
constraint_tight = binvar(T,2,num_rollout,'full');
M = 1e5;
grad_affine = cell(T,num_rollout);
grad_dyn = cell(T-1,num_rollout);
lambda_dyn = sdpvar(T-1, state_dim, num_rollout,'full');
affine_residual = cell(T, num_rollout);
slack = sdpvar(T,1);
dim = 1;

grad_term = cell(num_rollout,1);
grad_init = cell(num_rollout,1);
lambda_term = sdpvar(1,4,num_rollout,'full');
lambda_init = sdpvar(1,4,num_rollout,'full');


constrained_states_upper_limited = 4;
for rollout_cnt = 1:num_rollout
    for k = 1:T-1
        z_k = z{rollout_cnt}((k-1)*state_dim+1:k*state_dim);
        v_k = v{rollout_cnt}((k-1)*input_dim+1:k*input_dim);
        z_k_plus_1 = z{rollout_cnt}(k*state_dim+1:(k+1)*state_dim);
        dyn_residual = z_k_plus_1 - A_t*z_k - B_t*v_k;

        grad_dyn{k,rollout_cnt} = jacobian(dyn_residual,t_all);


        % constraints = [constraints, replace(dyn_residual, z{rollout_cnt}, state_trajectory_closedloop{rollout_cnt}(:)) == 0];
        
        % No robustifying terms
        affine_residual{k,rollout_cnt} = A_poly(dim,:)*z_k - b_poly(dim);
       

        % Primal feasibility
        constraints = [constraints, replace(affine_residual{k,rollout_cnt}, z{rollout_cnt}, state_trajectory_closedloop{rollout_cnt}(:)) <= 0];
        
        % Complementary slackness
        constraints = [constraints, replace(affine_residual{k,rollout_cnt},z{rollout_cnt},state_trajectory_closedloop{rollout_cnt}(:)) >= -M*constraint_tight(k,1)];
        constraints = [constraints, 0 <= lambda_affine(k,rollout_cnt), lambda_affine(k,rollout_cnt) <= M*constraint_tight(k,2,rollout_cnt)];
        constraints = [constraints, sum(constraint_tight(k,:,rollout_cnt))<=1];
        
        % Stationarity contribution
        grad_affine{k,rollout_cnt} = jacobian(affine_residual{k,rollout_cnt},t_all);

        % constraints = [constraints, [1 0]*v_k + norm([1 0]*phi_ku,1) <= 75];
        % constraints = [constraints, [1 0]*v_k - norm([1 0]*phi_ku,1) >= -75];
        % constraints = [constraints, [0 1]*v_k + norm([0 1]*phi_ku,1) <= 75];
        % constraints = [constraints, [0 1]*v_k - norm([0 1]*phi_ku,1) >= -75];
    end

    % Start and terminal constraint
    z_init = z{rollout_cnt}((1-1)*state_dim+1:1*state_dim);
    z_terminal = z{rollout_cnt}((T-1)*state_dim+1:T*state_dim);
    
    % Stationarity contribution of terminal constraints
    grad_term{rollout_cnt} = jacobian(z_terminal,t_all);
    grad_init{rollout_cnt} = jacobian(z_init,t_all);
end



% Terminal robust constraint
for rollout_cnt = 1:num_rollout
    z_terminal = z{rollout_cnt}((T-1)*state_dim+1:T*state_dim);
    affine_residual{T,rollout_cnt} = A_poly(dim,:)*z_terminal - b_poly(dim);

    % Primal feasibility
    constraints = [constraints, replace(affine_residual{T,rollout_cnt}, z{rollout_cnt}, state_trajectory_closedloop{rollout_cnt}(:)) <= 0];

    % Complementary slackness
    constraints = [constraints, replace(affine_residual{T,rollout_cnt}, z{rollout_cnt}, state_trajectory_closedloop{rollout_cnt}(:)) >= -M*constraint_tight(T,1,rollout_cnt)];
    constraints = [constraints, 0 <= lambda_affine(T,rollout_cnt), lambda_affine(T,rollout_cnt) <= M*constraint_tight(T,2,rollout_cnt)];
    constraints = [constraints, sum(constraint_tight(T,:,rollout_cnt))<=1];
    grad_affine{T,rollout_cnt} = jacobian(affine_residual{T,rollout_cnt},t_all);

end


% SLS constraints
% constraints = [constraints, [eye(T*state_dim) - Z*A, -Z*B]*[phi_x;phi_u] == eye(T*state_dim)];


% Objective
objective = 0;
for rollout_cnt = 1:num_rollout
    for k = 1:T-1
        z_k = z{rollout_cnt}((k-1)*state_dim+1:k*state_dim);
        z_k_plus_1 = z{rollout_cnt}(k*state_dim+1:(k+1)*state_dim);
        
        objective = objective + sum((z_k_plus_1(1:2) - z_k(1:2)).^2) - z_k(1);
    end
end

grad_obj = jacobian(objective, t_all);

% % dynamics jacobians
% bicycle_dyn_f = @(x) [x(1),x(2),x(3),x(4)]';
% DT = 0.1;
% bicycle_dyn_g = @(x,u) [0 0 1 0;0 0 0 1; 0 0 0 0; 0 0 0 0]*x + [0 0;0 0;1 0;0 1]*u;
% 
% 
% traj = unstacked_z;
% u_traj = unstacked_v;
% traj_var_sym = sym('traj_var', size(traj));
% u_traj_var_sym = sym('u_traj_var', size(u_traj));
% 
% nu_var_dyn_term = []; % the collection of constraints residual
% for i = 1:size(traj, 2)-1
% nu_var_dyn_term = [nu_var_dyn_term, traj_var_sym(:, i+1) - ( traj_var_sym(:, i) + ...
%   bicycle_dyn_g(traj_var_sym(:,i),u_traj_var_sym(:, i))*DT)];
% end
% nu_var_dyn_jac = jacobian(nu_var_dyn_term(:), [traj_var_sym(:); u_traj_var_sym(:)]);
% dyn_jacobians = double(subs(nu_var_dyn_jac, [traj_var_sym(:); u_traj_var_sym(:)], [traj(:); u_traj(:)]));


% Collect the stationarity terms
stationarity = 0;
for rollout_cnt = num_rollout
    for t = 1:T
        stationarity = stationarity + grad_affine{t,rollout_cnt} * lambda_affine(t,rollout_cnt) + grad_obj;
        if t<=19
            stationarity = stationarity + lambda_dyn(t,:,rollout_cnt) * grad_dyn{t,rollout_cnt} + lambda_term(:,:,rollout_cnt) * grad_term{rollout_cnt};
            stationarity = stationarity + lambda_init(:,:,rollout_cnt) * grad_init{rollout_cnt};
        end
    end
end
% nu_var_dyn = sdpvar(1, size(dyn_jacobians, 1)); %nx * T-1
% nu_dyn = nu_var_dyn * dyn_jacobians;
% stationarity = stationarity + nu_dyn;
% nu_var1 = sdpvar(1, 2*size(traj,1) );
% nu_term1 = [nu_var1(1:size(traj, 1) ), zeros(1, length(traj(:)) - 2*size(traj, 1)), ...
% nu_var1(size(traj, 1)+1:end ), zeros(1, length(u_traj(:)))];
% nu_term = nu_term1;
% stationarity = stationarity + grad_obj + nu_term;

% Plug in the solved nominal trajectory
% numerical_stationarity = replace(stationarity, z, z_norm(:));
numerical_stationarity = replace(stationarity, t_all, t_all_numerical);
% Solve
% constraints = [constraints, slack >= 0];

% Sanity checks
% constraints = [constraints, b_poly >= 3]; % This should give non-zero stationarity
% constraints = [constraints, b_poly <=2.45]; % This should set the problem to be infeasible

ops = sdpsettings('solver','gurobi','verbose', 2);
optimization_results = optimize(constraints, norm(numerical_stationarity,1), ops);


%%
% Plot nominal trajectory (z)
figure;hold on;
% unstacked_z = reshape(value(z),[state_dim, T]);
% unstacked_v = reshape(value(v),[input_dim, T]);
% plot(unstacked_z(1,:),unstacked_z(2,:),"g","DisplayName","Nominal");
% plot(unstacked_z(1,:) + value(tube_size),unstacked_z(2,:),'Color','m','DisplayName','Tube Bound',LineWidth=3)

xlim([-10,10]);ylim([-10,10]);
%% Run roll outs at multiple times with noise signal

for rollout_cnt = 1:num_rollout
    % Roll out trajectory with noise to the dynamics, WITHOUT the feedback
    x_init = [0;0;0;0]; % same as nominal trajectory initial state
    x = state_trajectory_closedloop{rollout_cnt};
    plot(x(1,1:end),x(2,1:end),"b","DisplayName","Disturbed w/ Feedback Control");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"b","DisplayName","Error signal w/ Feedback Control")
end
% xline(b_poly,":",'LineWidth',2);
save('forward_game_data.mat','state_trajectory_closedloop','input_trajectory_closedloop','unstacked_z','unstacked_v','state_dim','input_dim','DimAffine','T','num_rollout','val_phi_u','val_phi_x','Z','A','B','A_t','B_t', ...
    'disturbance_level');
xlabel('x',Interpreter='latex');
ylabel('y',Interpreter='latex');

hNom = plot(unstacked_z(1,1:end),unstacked_z(2,1:end), "g", "DisplayName","Nominal Trajectory",'LineWidth',3);
lineLearn = xline(value(b_poly),"--",'LineWidth',2,'Color','k','DisplayName','Learned Constraint(s)');
lineTruth = xline(5,'LineWidth',2,'Color','y','DisplayName','Ground Truth Constraint(s)');
hOpen = plot(nan, nan, 'r', 'DisplayName','Open-loop (disturbed)');
hFB   = plot(nan, nan, 'b', 'DisplayName','Feedback (disturbed)');
hStart = plot(unstacked_z(1,1), unstacked_z(2,1), 'o', 'LineStyle','none', ...
    'MarkerSize',8, 'MarkerFaceColor','g', 'MarkerEdgeColor','k', ...
    'DisplayName','Start');

hGoal  = plot(unstacked_z(1,end), unstacked_z(2,end), 'p', 'LineStyle','none', ...
    'MarkerSize',11, 'MarkerFaceColor','g', 'MarkerEdgeColor','k', ...
    'DisplayName','Goal');

% legend([hNom hOpen hFB lineLearn lineTruth], 'Interpreter','latex', 'Location','best');
legend([hNom hFB lineLearn lineTruth], 'Interpreter','latex', 'Location','best');


%% 
% 1) Axes cosmetics & LaTeX
ax = gca; box on; grid on; ax.Layer = 'top';
ax.TickDir = 'out'; ax.LineWidth = 1;
ax.FontName = 'Times New Roman'; ax.FontSize = 10;  % or your journal’s font
set(ax,'TickLabelInterpreter','latex');
axis equal;xlim([-10,20]);ylim([-2,20]);


% 2) Shade the forbidden half-space (to the right of the constraint line)
yl = ylim; xr = xlim;
hForbid = patch([value(b_poly) xr(2) xr(2) value(b_poly)], [yl(1) yl(1) yl(2) yl(2)], ...
    [0 0 0], 'FaceAlpha',0.06, 'EdgeColor','none', 'HandleVisibility','off');
uistack(hForbid,'bottom');  % keep it behind the trajectories
text(value(b_poly) + 4, mean(yl), '\textbf{unsafe}', 'Interpreter','latex', ...
     'Rotation',45, 'HorizontalAlignment','left', 'Color',[0 0 0],'FontSize',30);

