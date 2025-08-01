%% Load the phi matrices
load("phi_matrices_inverse_game.mat");
load("v_nominal_val.mat");
load("z_nominal_val.mat");

%% KKT conditions 
M = 1;
N = 4;       % state dimension
Nu = 2;      % control dimension
T = 10;
dimAffine = 4;

U_MAX = 2;
L_MAX = 100;
M_big = 1e5;
epsilon = 1e-4;

state_dim = 4;
input_dim = 2;
output_dim = 2;

% Prepare trajectory vars
traj_mat = {z_nominal_val};
u_traj_mat = {v_nominal_val};
traj_var = cell(M,1);
u_traj_var = cell(M,1);
t_all_vec = [];
% 
xl = sdpvar(1,1);
xu = sdpvar(1,1);
yl = sdpvar(1,1);
yu = sdpvar(1,1);

C_t = [1 0 0 0;
       0 1 0 0];

goal = [4,0,0,0]';


%% Populate the symbolic variables

dyn_noise_magnitude = 0.05;
obs_noise_magbitude = 0.02;
for m = 1:M
    traj_var{m} = sdpvar(N, T, 'full');
    u_traj_var{m} = sdpvar(Nu, T, 'full');
    t_all_vec = [t_all_vec, traj_var{m}(:)', u_traj_var{m}(:)'];
end

stationarity = zeros(size(t_all_vec));
constraints = [];

%% Calculate the jacobians

doubleint_dyn_f = @(x) [x(1),x(2),x(3),x(4)]';
DT = 0.5;
doubleint_dyn_g = @(x,u) [0 0 1 0;0 0 0 1; 0 0 0 0; 0 0 0 0]*x*DT + [0 0;0 0;1 0;0 1]*u*DT;

traj_var_sym = sym('traj_var', size(traj_mat{1}));
u_traj_var_sym = sym('u_traj_var', size(u_traj_mat{1}));

nu_var_dyn_term = []; % the collection of constraints residual
for i = 1:T-1
nu_var_dyn_term = [nu_var_dyn_term, traj_var_sym(:, i+1) - ( traj_var_sym(:, i) + ...
  doubleint_dyn_g(traj_var_sym(:,i),u_traj_var_sym(:, i)))];
end
nu_var_dyn_jac = jacobian(nu_var_dyn_term(:), [traj_var_sym(:); u_traj_var_sym(:)]);
dyn_jacobians = {double(subs(nu_var_dyn_jac, [traj_var_sym(:); u_traj_var_sym(:)], [traj_mat{1}(:); u_traj_mat{1}(:)]))};

%% Control constraint
% TODO: This should be robustified
lambda_u = cell(M,1);
g_u = cell(M,1);
g_u_grad = cell(M,1);
input_tube_across_time = zeros(input_dim,T);

for m = 1:M
    % 4 input constraints at each timestep
    lambda_u{m} = sdpvar(4, T,'full');

    for i = 1:T
        robust_tube = zeros(input_dim,1);
        % Compute the robust tube
        for j = 1:i
            phi_uw_blk_ij = phi_uw_inv(input_dim*(i-1)+1:input_dim*i,state_dim*(j-1)+1:state_dim*j);
            phi_ue_blk_ij = phi_ue_inv(input_dim*(i-1)+1:input_dim*i,output_dim*(j-1)+1:output_dim*j);
            robust_tube = robust_tube + dyn_noise_magnitude*sum(abs(phi_uw_blk_ij),2) + obs_noise_magbitude*sum(abs(phi_ue_blk_ij),2);
        end
        input_tube_across_time(:,i) = robust_tube;
        % g_temp = sum(u_traj_var{m}.^2, 1) - U_MAX;
        % g_u{m} = replace(g_temp, u_traj_var{m}, u_traj_mat{m});
        % g_grad_temp = jacobian(g_temp', t_all_vec);
        % g_u_grad{m} = replace(g_grad_temp, u_traj_var{m}, u_traj_mat{m});
        g_u_1 = u_traj_var{m}(1,i) + robust_tube(1) - U_MAX;
        g_u_2 = u_traj_var{m}(2,i) + robust_tube(2) - U_MAX;
        g_u_3 = -U_MAX - (u_traj_var{m}(1,i) - robust_tube(1));
        g_u_4 = -U_MAX - (u_traj_var{m}(2,i) - robust_tube(2));
        g_u_1_grad = jacobian(g_u_1, t_all_vec);
        g_u_2_grad = jacobian(g_u_2, t_all_vec);
        g_u_3_grad = jacobian(g_u_3, t_all_vec);
        g_u_4_grad = jacobian(g_u_4, t_all_vec);
        stationarity = stationarity + lambda_u{m}(1,i) * replace(g_u_1_grad,u_traj_var{m},u_traj_mat{m});
        stationarity = stationarity + lambda_u{m}(2,i) * replace(g_u_2_grad,u_traj_var{m},u_traj_mat{m});
        stationarity = stationarity + lambda_u{m}(3,i) * replace(g_u_3_grad,u_traj_var{m},u_traj_mat{m});
        stationarity = stationarity + lambda_u{m}(4,i) * replace(g_u_4_grad,u_traj_var{m},u_traj_mat{m});
        constraints = [constraints, -epsilon <= replace(g_u_1,u_traj_var{m},u_traj_mat{m}) * lambda_u{m}(1,i) <= epsilon, lambda_u{m}(1,i) >= 0];
        constraints = [constraints, -epsilon <= replace(g_u_2,u_traj_var{m},u_traj_mat{m}) * lambda_u{m}(2,i) <= epsilon, lambda_u{m}(2,i) >= 0];
        constraints = [constraints, -epsilon <= replace(g_u_3,u_traj_var{m},u_traj_mat{m}) * lambda_u{m}(3,i) <= epsilon, lambda_u{m}(3,i) >= 0];
        constraints = [constraints, -epsilon <= replace(g_u_4,u_traj_var{m},u_traj_mat{m}) * lambda_u{m}(4,i) <= epsilon, lambda_u{m}(4,i) >= 0];
    end
end

%% Affine box constraints (no `obs`)
% TODO: This should be robustified
lambda_affine = sdpvar(M, dimAffine, T, 'full');
g_affine = sdpvar(M, dimAffine, T, 'full');
g_affine_grad = sdpvar(M, dimAffine, T, length(t_all_vec), 'full');
output_tube_across_time = zeros(output_dim,T);


for m = 1:M
    for i = 1:T
        robust_tube = zeros(output_dim,1);
        for j = 1:i
            phi_xw_blk_ij = phi_xw_inv(state_dim*(i-1)+1:state_dim*i,state_dim*(j-1)+1:state_dim*j);
            phi_xe_blk_ij = phi_xe_inv(state_dim*(i-1)+1:state_dim*i,output_dim*(j-1)+1:output_dim*j);
            robust_tube = robust_tube + dyn_noise_magnitude*sum(abs(C_t * phi_xw_blk_ij),2) + obs_noise_magbitude*sum(abs(C_t * phi_xe_blk_ij),2);
        end
        output_tube_across_time(:,i) = robust_tube;
        g_affine(m,1,i) = -(traj_var{m}(1,i) - robust_tube(1) - xu);
        g_affine(m,2,i) = -(traj_var{m}(2,i) - robust_tube(2) - yu);
        g_affine(m,3,i) = -(-xl - traj_var{m}(1,i) - robust_tube(1));
        g_affine(m,4,i) = -(-yl - traj_var{m}(2,i) - robust_tube(2));
    end
    for j = 1:dimAffine
        g = reshape(g_affine(m,j,:), [], 1);
        g_grad = jacobian(g, t_all_vec);
        g_affine(m,j,:) = reshape(replace(g, traj_var{m}, traj_mat{m}), 1, T);
        g_affine_grad(m,j,:,:) = replace(g_grad, traj_var{m}, traj_mat{m});
    end
end

%% Dynamics-related terms
nu_term = cell(M,1);
nu_dyn = cell(M,1);
nu_dyn_con = [];
nu_term_con = [];

for m = 1:M
    nu_dyn_var = sdpvar(1, size(dyn_jacobians{m}, 1));
    nu_dyn{m} = nu_dyn_var * dyn_jacobians{m};
    nu_dyn_con = [nu_dyn_con, nu_dyn{m}];

    nu_temp = sdpvar(1, N + 2);
    % nu_term{m} = [nu_temp(1:N), zeros(1, numel(traj_var{m}) - 2*N), ...
    %               nu_temp(N+1:N+2), 0, 0, zeros(1, numel(u_traj_var{m}))];
    nu_term{m} = [nu_temp(1:N), zeros(1, numel(traj_var{m}) - 2*N), ...
                  0, 0, 0, 0, zeros(1, numel(u_traj_var{m}))];
    nu_term_con = [nu_term_con, nu_term{m}];
end

%% Objective gradient (trajectory smoothness)
% TODO: This should be replaced by the actual objectives
jac_manual = cell(M,1);
for m = 1:M
    obj = sum(sum((traj_var{m}(1:2,2:end) - traj_var{m}(1:2,1:end-1)).^2));
    for i = 1:T
        obj = obj + 1/T * norm(traj_var{m}(1:2,i)-goal(1:2))^2;
    end
    obj = obj + 1e3 * norm(traj_var{m}(1:2,end)-goal(1:2))^2;
    jac_manual{m} = replace(jacobian(obj, t_all_vec), traj_var{m}, traj_mat{m});
end

%% Complementary slackness terms (no obs dim)
z = binvar(M, dimAffine, T, 2, 'full');
q = binvar(M, dimAffine, T, 'full');
which_halfspace = binvar(M, dimAffine, T, 'full');
r = cell(M, dimAffine);
q_rep = cell(M,dimAffine);


for m = 1:M
    for j = 1:dimAffine
        grad_vec = reshape(g_affine_grad(m,j,:,:), T, length(t_all_vec));
        r{m,j} = sdpvar(1, size(grad_vec,2));
        q_vec = reshape(q(m,j,:),1,[]);
        
        if m ==1
            q_state = [q_vec;q_vec;q_vec;q_vec];
            q_u = [zeros(1,T);zeros(1,T)];
            another_agent_q_state = [zeros(1,T) ;zeros(1,T) ;zeros(1,T) ;zeros(1,T) ];
            another_agent_q_u = [zeros(1,T-1);zeros(1,T-1)];
            % q_rep{m,j} = [q_state(:)',q_u(:)',q_state(:)',q_u(:)'];
            q_rep{m,j} = [q_state(:)',q_u(:)'];
        else
            q_state = [q_vec;q_vec ;q_vec ;q_vec];
            q_u = [zeros(1,T-1);zeros(1,T-1)];
            another_agent_q_state = [zeros(1,T) ;zeros(1,T) ;zeros(1,T) ;zeros(1,T) ];
            another_agent_q_u = [zeros(1,T-1);zeros(1,T-1)];
            q_rep{m,j} = [q_state(:)',q_u(:)',q_state(:)',q_u(:)'];
        end
        % 
        % q_state = [q_vec;q_vec;q_vec;q_vec];
        % q_u = [zeros(1,T-1);zeros(1,T-1)];
        % q_rep = [q_state(:)',q_u(:)',q_state(:)',q_u(:)'];

        for i = 1:T
        constraints = [constraints, ...
            lambda_affine(m,j,i) <= M_big*z(m,j,i,1), ...
            -g_affine(m,j,i) <= M_big*z(m,j,i,2), ...
            z(m,j,i,1) + z(m,j,i,2) <= 2 - q(m,j,i), ...
            g_affine(m,j,i) <= M_big*(1-q(m,j,i)), ...
            0 <= lambda_affine(m,j,i) <= L_MAX];
        end

        constraints = [constraints, ...
                        r{m,j} >= reshape(lambda_affine(m,j,:),1,[]) * grad_vec - (1 - q_rep{m,j}) * M_big, ...
                        r{m,j} <= reshape(lambda_affine(m,j,:),1,[]) * grad_vec + (1 - q_rep{m,j}) * M_big, ...
                        -M_big*q_rep{m,j} <= r{m,j} <= M_big*q_rep{m,j}, ...
                        -M_big <= r{m,j} <= M_big];
    end
end

for m  = 1:M
    for i = 1:T
        constraints = [constraints, sum(which_halfspace(m,:,i)) >= 1];
    end
end

% for m = 1:M
%     constraints = [constraints, -epsilon <= lambda_u{m} .* g_u{m} <= epsilon, lambda_u{m} >= 0 ];
% end

% constraints = [constraints, xu>=xl, yu>=yl];
constraints = [constraints, xu>=0, yu>=0, xl>=0, yl>=0];
% Sanity check
% constraints = [constraints, xu==3,yu==1,xl==3,yl==1];
%% Final stationarity condition
stationarity = stationarity + nu_term_con + nu_dyn_con;
for m = 1:M
    stationarity = stationarity + jac_manual{m};
    for j = 1:dimAffine
            % grad_vec = reshape(g_affine_grad(m,j,i,:), 1, []);
            % stationarity = stationarity + lambda_affine(m,j,i) * grad_vec * q(m,j,i);
        stationarity = stationarity + r{m,j};
    end
end

%% Optimization
ops = sdpsettings('solver','gurobi','verbose', 2);
optimization_results = optimize(constraints, norm(stationarity,1),ops);