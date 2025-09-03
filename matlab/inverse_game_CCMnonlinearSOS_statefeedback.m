%% Skip the K inverse and phi inverse for now as they are trivial
%% Write the KKT conditions for robust-constraints-learning

constraints = [];
% Nominal trajectory z,v
z = sdpvar(state_dim*T,1);
v = sdpvar(input_dim*T,1);
% A_poly = sdpvar(1,4);
A_poly = [1 0 0 0];
b_poly = sdpvar(1,1);

lambda_affine = sdpvar(T,1);
constraint_tight = binvar(T,2);
M = 1e5;
grad_affine = cell(T,1);
grad_dyn = cell(T-1,1);
lambda_dyn = sdpvar(T-1, state_dim,'full');
affine_residual = cell(T,1);
slack = sdpvar(T,1);

phi_x = sol.value(phi_x);
phi_u = sol.value(phi_u);

constrained_states_upper_limited = 4;
for dim = 1:DimAffine
    for k = 1:T-1
        phi_kx = phi_x((k-1)*state_dim+1:k*state_dim,:);
        phi_ku = phi_u((k-1)*input_dim+1:k*input_dim,:);
        z_k = z((k-1)*state_dim+1:k*state_dim);
        v_k = v((k-1)*input_dim+1:k*input_dim);
        z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
        % dyn_residual = z_k_plus_1 - f(z_k) - B_t*v_k;
        % 
        % grad_dyn{k} = jacobian(dyn_residual,z);

 
        % constraints = [constraints, replace(dyn_residual,z,unstacked_z(:)) == 0];
        robust_affine_constraints = 0;
        for j = 1:k
            robust_affine_constraints = robust_affine_constraints + ...
                disturbance_level * norm(A_poly(dim,:)*phi_kx(:,(j-1)*state_dim+1:j*state_dim),1);
        end
        % constraints = [constraints, A_poly(dim,:)*z_k + robust_affine_constraints <= b_poly(dim)];
        affine_residual{k} = A_poly(dim,:)*z_k + robust_affine_constraints - b_poly(dim);
       

        % Primal feasibility
        constraints = [constraints, replace(affine_residual{k},z,unstacked_z(:)) <= 0];
        
        % Complementary slackness
        constraints = [constraints, replace(affine_residual{k},z,unstacked_z(:)) >= -M*constraint_tight(k,1)];
        constraints = [constraints, 0 <= lambda_affine(k), lambda_affine(k) <= M*constraint_tight(k,2)];
        constraints = [constraints, sum(constraint_tight(k,:))<=1];
        
        % Stationarity contribution
        grad_affine{k} = jacobian(affine_residual{k},[z;v]);

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

% In the inverse game we dont know the exact terminal/initial constraints
% constraints = [constraints, z_init(1:state_dim) == [0;0;0;0]];
% constraints = [constraints, z_terminal(1:state_dim) == [2.9;20;0;0]];
% constraints = [constraints, v_terminal(1:input_dim) == [0;0]];



% Stationarity contribution of terminal constraints
grad_term = jacobian(z_terminal,z);
grad_init = jacobian(z_init,z);
lambda_term = sdpvar(1,4);
lambda_init = sdpvar(1,4);


% Terminal robust constraint
for dim = 1:DimAffine
    robust_affine_constraints = 0;
    for j = 1:T
        robust_affine_constraints = robust_affine_constraints + disturbance_level * ...
            norm(A_poly(dim,:)*phi_x((T-1)*state_dim+1:T*state_dim,(j-1)*state_dim+1:j*state_dim),1);
    end
    affine_residual{T} = A_poly(dim,:)*z_terminal + robust_affine_constraints - b_poly(dim);

    % Primal feasibility
    constraints = [constraints, replace(affine_residual{T},z,unstacked_z(:)) <= 0];

    % Complementary slackness
    constraints = [constraints, replace(affine_residual{T},z,unstacked_z(:)) >= -M*constraint_tight(T,1)];
    constraints = [constraints, 0 <= lambda_affine(T), lambda_affine(T) <= M*constraint_tight(T,2)];
    constraints = [constraints, sum(constraint_tight(T,:))<=1];
    grad_affine{T} = jacobian(affine_residual{T},[z;v]);

end


% SLS constraints
% constraints = [constraints, [eye(T*state_dim) - Z*A, -Z*B]*[phi_x;phi_u] == eye(T*state_dim)];


% Objective
objective = 0;
for k = 1:T-1
    z_k = z((k-1)*state_dim+1:k*state_dim);
    z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
    
    objective = objective + sum((z_k_plus_1(1:2) - z_k(1:2)).^2) - z_k(1);
end

grad_obj = jacobian(objective, [z;v]);

% dynamics jacobians
% bicycle_dyn_f = @(x) [x(1),x(2),x(3),x(4)]';

bicycle_dyn_f = @(x) [x(4)*cos(x(3)),x(4)*sin(x(3)),0,0]';
DT = 0.2;
bicycle_dyn_g = @(x,u) [0 0;0 0;1 0;0 1]*u;


traj = unstacked_z;
u_traj = unstacked_v;
traj_var_sym = sym('traj_var', size(traj));
u_traj_var_sym = sym('u_traj_var', size(u_traj));

nu_var_dyn_term = []; % the collection of constraints residual
for i = 1:size(traj, 2)-1
nu_var_dyn_term = [nu_var_dyn_term, traj_var_sym(:, i+1) - ( traj_var_sym(:, i) + ...
  bicycle_dyn_g(traj_var_sym(:,i),u_traj_var_sym(:, i))*DT + bicycle_dyn_f(traj_var_sym(:,i))*DT)];
end
nu_var_dyn_jac = jacobian(nu_var_dyn_term(:), [traj_var_sym(:); u_traj_var_sym(:)]);
% dyn_jacobians = replace(nu_var_dyn_jac, [traj_var_sym(:); u_traj_var_sym(:)], [traj(:); u_traj(:)]);
dyn_jacobians = double(subs(nu_var_dyn_jac, [traj_var_sym(:); u_traj_var_sym(:)], [traj(:); u_traj(:)]));


% Collect the stationarity terms
stationarity = 0;
for t = 1:T
    stationarity = stationarity + grad_affine{t} * lambda_affine(t); 
end
nu_var_dyn = sdpvar(1, size(dyn_jacobians, 1)); %nx * T-1
nu_dyn = nu_var_dyn * dyn_jacobians;
stationarity = stationarity + nu_dyn;
nu_var1 = sdpvar(1, 2*size(traj,1) );
nu_term1 = [nu_var1(1:size(traj, 1) ), zeros(1, length(traj(:)) - 2*size(traj, 1)), ...
nu_var1(size(traj, 1)+1:end ), zeros(1, length(u_traj(:)))];
nu_term = nu_term1;
stationarity = stationarity + grad_obj + nu_term;

% Plug in the solved nominal trajectory
% numerical_stationarity = replace(stationarity, z, z_norm(:));
numerical_stationarity = 0.01*replace(stationarity, z, unstacked_z(:));
% Solve
constraints = [constraints, slack >= 0];

% Sanity checks
% constraints = [constraints, b_poly <= 4.99]; % This should give non-zero stationarity
% constraints = [constraints, b_poly >= 5.01]; % This should set the problem to be infeasible

ops = sdpsettings('solver','gurobi','verbose', 2);
optimization_results = optimize(constraints, norm(numerical_stationarity,1), ops);