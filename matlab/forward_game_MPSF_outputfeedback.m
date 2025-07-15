clc;clear;close all;

T = 20; % Time horizon
Ts = 0.1; % Sampling time
state_dim = 4;
input_dim = 2;
output_dim = 2;
disturbance_level_dyn = 0; % The magnitude of disturbances in dynamics
disturbance_level_output = 1; % The magnitude of disturbances in sensor

% Double integrator dynamics
A_t = [0 0 1 0;0 0 0 1; 0 0 0 0; 0 0 0 0]*Ts+ eye(state_dim);
% B_t = [1/2*Ts^2 0;0 1/2*Ts^2;Ts 0;0 Ts]; 
% Omit the second order terms
B_t = [0 0;0 0;1 0;0 1]*Ts;
C_t = [1 0 0 0;
       0 1 0 0];

num_repetitions_A = T - 1;
num_repetitions_B = T - 1;
num_repetitions_C = T - 1;
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

matrices_C = cell(1, num_repetitions_C);
for i = 1:num_repetitions_C+1
    if i == T
        matrices_C{i} = C_t + 0;
    else
        matrices_C{i} = C_t;
    end
end

A = blkdiag(matrices_A{:});
B = blkdiag(matrices_B{:});
C = blkdiag(matrices_C{:});

Z = createBlockDownshiftOperator(state_dim,T);

% Construct the phi block matrices
phi_xw = sdpvar(T*state_dim, T*state_dim,'full');
phi_uw = sdpvar(T*input_dim, T*state_dim,'full');

phi_xe = sdpvar(T*state_dim, T*output_dim,'full');
phi_ue = sdpvar(T*input_dim, T*output_dim,'full');
% 
% phi_yw = sdpvar(T*output_dim, T*state_dim,'full');
% phi_ye = sdpvar(T*output_dim, T*output_dim,'full');

constraints = [];

% Populate the phis as blcok lower triangular matrices
for i = 1:T
    for j = 1:T
        if j>i
            constraints = [constraints,  phi_xw((i-1)*state_dim+1:i*state_dim,(j-1)*state_dim+1:j*state_dim) == zeros(state_dim,state_dim)];
            constraints = [constraints,  phi_uw((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) == zeros(input_dim,state_dim)];
            constraints = [constraints,  phi_xe((i-1)*state_dim+1:i*state_dim,(j-1)*output_dim+1:j*output_dim) == zeros(state_dim,output_dim)];
            constraints = [constraints,  phi_ue((i-1)*input_dim+1:i*input_dim,(j-1)*output_dim+1:j*output_dim) == zeros(input_dim,output_dim)];
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
         0 -1 0 0 ];
b_poly = [5;
        5;
        2;  
        2];
DimAffine = 1;

% Nominal trajectory z,v,y
z = sdpvar(state_dim*T,1);
v = sdpvar(input_dim*T,1);
y = sdpvar(output_dim*T,1);
constraint_value = {};
robust_terms = {};
for dim = 1:DimAffine
    for k = 1:T-1
        phi_kxw = phi_xw((k-1)*state_dim+1:k*state_dim,:);
        phi_kxe = phi_xe((k-1)*state_dim+1:k*state_dim,:);
        z_k = z((k-1)*state_dim+1:k*state_dim);
        v_k = v((k-1)*input_dim+1:k*input_dim);
        z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
        y_k = C_t * z_k;
        constraints = [constraints, z_k_plus_1 == A_t*z_k + B_t*v_k];
        robust_affine_constraints = 0;
        for j = 1:k
            robust_affine_constraints = robust_affine_constraints +  ...
                disturbance_level_dyn * norm(phi_kxw(:,(j-1)*state_dim+1:j*state_dim),1) + ...
                disturbance_level_output * norm(phi_kxe(:,(j-1)*output_dim+1:j*output_dim),1);
        end
        
        % Assume constraints are only on output (i.e. what the agent senses)
        constraints = [constraints, z_k(1) + robust_affine_constraints(1) <= b_poly(dim)];
        constraint_value{k} = z_k + robust_affine_constraints - b_poly(dim);
        robust_terms{k} = robust_affine_constraints;
        % constraints = [constraints, [1 0]*v_k + norm([1 0]*phi_ku,1) <= 75];
        % constraints = [constraints, [1 0]*v_k - norm([1 0]*phi_ku,1) >= -75];
        % constraints = [constraints, [0 1]*v_k + norm([0 1]*phi_ku,1) <= 75];
        % constraints = [constraints, [0 1]*v_k - norm([0 1]*phi_ku,1) >= -75];
    end
end

% Start and terminal constraint
z_init = z((1-1)*state_dim+1:1*state_dim);
z_terminal = z((T-1)*state_dim+1:T*state_dim);
y_terminal = C_t * z_terminal;
% v_terminal = v((T-1)*input_dim+1:T*input_dim);
constraints = [constraints, z_init(1:state_dim) == [0;0;0;0]];
constraints = [constraints, z_terminal(1:state_dim) == [0.;20;0;0]];
% constraints = [constraints, v_terminal(1:input_dim) == [0;0]];


% Terminal robust constraint
for dim = 1:DimAffine
    robust_affine_constraints = 0;
    for j = 1:T
        robust_affine_constraints = robust_affine_constraints +  ...
            disturbance_level_dyn *  norm(phi_xw((T-1)*state_dim+1:T*state_dim,(j-1)*state_dim+1:j*state_dim),1) + ...
            disturbance_level_output *  norm(phi_xe((T-1)*state_dim+1:T*state_dim,(j-1)*output_dim+1:j*output_dim),1);
    end
    constraints = [constraints, z_terminal(1) + robust_affine_constraints(1) <= b_poly(dim)];
end


% % SLS constraints (frank's framework)
% constraints = [constraints, [eye(T*state_dim) - Z*A, -Z*B, zeros(T*state_dim,T*input_dim);  ...
%                              -C, zeros(T*output_dim,T*input_dim),eye(T*output_dim)]*[phi_xw, phi_xe;phi_uw,phi_ue;phi_yw,phi_ye] == eye(T*state_dim + T*output_dim)];

% SLS constraints (framework in iSLS paper)
constraints = [constraints, [eye(T*state_dim) - Z*A, -Z*B] * [phi_xw, phi_xe;phi_uw,phi_ue] == [eye(T*state_dim), zeros(T*state_dim,T*output_dim)]];
constraints = [constraints, [phi_xw, phi_xe;phi_uw,phi_ue] * [eye(T*state_dim) - Z*A; -C] == [eye(T*state_dim);zeros(T*input_dim,T*state_dim)]];

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
output_trajectory_closedloop = cell(num_rollout,1);
input_trajectory_closedloop = cell(num_rollout,1);





for rollout_cnt = 1:num_rollout
    % Roll out trajectory with noise to the dynamics, WITHOUT the feedback
    x_init = [0;0;0;0]; % same as nominal trajectory initial state
    x = zeros(state_dim,T);
    
    for i = 1:T-1
        % noise = randn(4, 1);     % random vector from N(0,1)
        noise = disturbance_level_dyn * (rand(4,1)*2 - 1);
        new_x = A_t*x(:,i) + B_t*unstacked_v(:,i) + noise;
        x(:,i+1) = new_x;
    end
    % plot(x(1,:),x(2,:),"r","DisplayName","Disturbed");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"r","DisplayName","Error signal without Feedback Control")
    error_signal_openloop{rollout_cnt} = [x(1,1:end) - unstacked_z(1,1:end);x(2,1:end) - unstacked_z(2,1:end)];
    
    % Roll out trajectory with noise to the dynamics, WITH the feedback
    
    % Suppress the NaNs
    val_phi_ue = value(phi_ue);
    val_phi_ue(isnan(val_phi_ue)) = 0;
    val_phi_uw = value(phi_uw);
    val_phi_uw(isnan(val_phi_uw)) = 0;
    val_phi_xe = value(phi_xe);
    val_phi_xw = value(phi_xw);
    % val_phi_yw = value(phi_yw);
    % val_phi_ye = value(phi_ye);
    
    
    K = val_phi_ue - val_phi_uw*inv(val_phi_xw)*val_phi_xe;
    
    x_init = disturbance_level_dyn * (rand(4,1)*2 - 1);
    x = zeros(state_dim,T);
    x(:,1) = x_init;
    feedback_u = zeros(input_dim,T);
    
    for i = 1:T
        feedback_control = zeros(input_dim,1);
        for j = 1:i
            observation_noise = disturbance_level_output * (rand(output_dim,1)*2 - 1);
            % Feedback gain exists for all previous timesteps (causal system)
            feedback_control = feedback_control + K((i-1)*input_dim+1:i*input_dim,(j-1)*output_dim+1:j*output_dim) * (C_t * x(:,j) - C_t * unstacked_z(:,j)+ observation_noise)  ;
        end

        % noise = randn(4, 1);     % random vector from N(0,1)
        % noise = noise / norm(noise,1);     % normalize to have norm 1
        dynamics_noise = disturbance_level_dyn * (rand(state_dim,1)*2 - 1);
        if i<=T-1
            x(:,i+1) = A_t*x(:,i) + B_t*(feedback_control + unstacked_v(:,i)) + dynamics_noise;
        end
        feedback_u(:,i) = feedback_control;
    end
    plot(x(1,1:end),x(2,1:end),"b","DisplayName","Disturbed w/ Feedback Control");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"b","DisplayName","Error signal w/ Feedback Control")
    error_signal_feedback{rollout_cnt} = [x(1,1:end) - unstacked_z(1,1:end);x(2,1:end) - unstacked_z(2,1:end)];
    output_trajectory_closedloop{rollout_cnt} = x;
    input_trajectory_closedloop{rollout_cnt} = feedback_u;
end
xline(b_poly(1),":",'LineWidth',2);
% save('forward_game_data.mat','output_trajectory_closedloop','input_trajectory_closedloop','unstacked_z','unstacked_v','state_dim','input_dim','DimAffine','T','num_rollout','val_phi_u','val_phi_x','Z','A','B','A_t','B_t', ...
%     'disturbance_level');
xlabel('x',Interpreter='latex');
ylabel('y',Interpreter='latex');

figure(2);
plot(unstacked_z(1,1:end),unstacked_z(2,1:end), "g", "DisplayName","Nominal Trajectory",'LineWidth',3);
xlabel('x',Interpreter='latex');
ylabel('y',Interpreter='latex');
xline(b_poly(1),":",'LineWidth',2);
xlim([-20,20]);ylim([-20,20]);

function D = createBlockDownshiftOperator(n_block, N)
% CREATEBLOCKDOWNSHIFTOPERATOR Creates a block downshift operator matrix.
%
%   D = CREATEBLOCKDOWNSHIFTOPERATOR(n_block, N) generates a matrix D that,
%   when multiplied by a stacked vector X of N blocks (each of size n_block),
%   shifts the blocks downwards. The operation is:
%   X_shifted = D * X
%   where X_shifted will have the first block replaced by zeros, and the
%   subsequent blocks shifted up, with a zero block at the end.
%
%   Inputs:
%     n_block - The dimension (size) of each individual block vector.
%     N       - The total number of blocks in the stacked vector.
%
%   Output:
%     D       - The (N*n_block) x (N*n_block) block downshift operator matrix.

    % Total dimension of the stacked vector
    total_dim = N * n_block;

    % Initialize the matrix D with zeros. Using sparse matrix for efficiency
    % if N*n_block is large, as most elements will be zero.
    D = sparse(total_dim, total_dim);

    % Populate the matrix.
    % The i-th block of the output vector (X_shifted) should be the (i+1)-th
    % block of the input vector (X).
    % This means we are mapping rows corresponding to block i in D to columns
    % corresponding to block i+1 in the input.

    for i = 1:(N - 1)
        % Define the row indices for the current output block (i-th block of X_shifted)
        row_start = i * n_block + 1;
        row_end = (i + 1) * n_block;

        % Define the column indices for the corresponding input block ((i+1)-th block of X)
        col_start = (i - 1) * n_block + 1;
        col_end = i * n_block;

        % Place an identity matrix in the appropriate block position
        D(row_start:row_end, col_start:col_end) = eye(n_block);
    end

    % For the last block of the output (N-th block of X_shifted), it will be
    % filled with zeros by default from the sparse initialization.
end