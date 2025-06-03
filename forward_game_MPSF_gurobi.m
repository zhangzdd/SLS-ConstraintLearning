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
            % constraints = [constraints,  phi_x((i-1)*state_dim+1:i*state_dim,(j-1)*state_dim+1:j*state_dim) == zeros(state_dim,state_dim)];
            % constraints = [constraints,  phi_u((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) == zeros(input_dim,state_dim)];
            phi_x((i-1)*state_dim+1:i*state_dim,(j-1)*state_dim+1:j*state_dim) = zeros(state_dim,state_dim);
            phi_u((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) = zeros(input_dim,state_dim);
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
b_poly = [3;
        3;
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
        constraints = [constraints, [1 0 0 0]*z_k + disturbance_level * norm([1 0 0 0]*phi_kx,1) <= 3];
        constraints = [constraints, [1 0]*v_k + norm([1 0]*phi_ku,1) <= 75];
        constraints = [constraints, [1 0]*v_k + norm([1 0]*phi_ku,1) >= -75];
        constraints = [constraints, [0 1]*v_k + norm([0 1]*phi_ku,1) <= 75];
        constraints = [constraints, [0 1]*v_k + norm([0 1]*phi_ku,1) >= -75];
    end
end

% Start and terminal constraint
z_init = z((1-1)*state_dim+1:1*state_dim);
z_terminal = z((T-1)*state_dim+1:T*state_dim);
v_terminal = v((T-1)*input_dim+1:T*input_dim);
constraints = [constraints, z_init(1:state_dim) == [0;0;0;0]];
constraints = [constraints, z_terminal(1:state_dim) == [2.95;9;0;0]];
constraints = [constraints, v_terminal(1:input_dim) == [0;0]];

% SLS constraints
constraints = [constraints, [eye(T*state_dim) - Z*A, -Z*B]*[phi_x;phi_u] == eye(T*state_dim)];


% Objective
objective = 0;
for k = 1:T-1
    z_k = z((k-1)*state_dim+1:k*state_dim);
    z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
    
    objective = objective + sum((z_k_plus_1(1:2) - z_k(1:2)).^2);
end

% Solve
ops = sdpsettings('solver','gurobi','verbose', 2);
optimization_results = optimize(constraints, objective, ops);
%%
% Plot nominal trajectory (z)
figure;hold on;
unstacked_z = reshape(value(z),[state_dim, T]);
unstacked_v = reshape(value(v),[input_dim, T]);
plot(unstacked_z(1,:),unstacked_z(2,:),"g","DisplayName","Nominal");
xlim([0,10]);ylim([0,10]);
%%
% Roll out trajectory with noise to the dynamics, WITHOUT the feedback
x_init = [0;0;0;0]; % same as nominal trajectory initial state
x = zeros(state_dim,T);

for i = 1:T-1
    noise = randn(4, 1);     % random vector from N(0,1)
    noise = noise / norm(noise);     % normalize to have norm 1
    new_x = A_t*x(:,i) + B_t*unstacked_v(:,i) + noise;
    x(:,i+1) = new_x;
end
plot(x(1,:),x(2,:),"r","DisplayName","Disturbed");
%%
% Roll out trajectory with noise to the dynamics, WITH the feedback

% Suppress the NaNs
val_u = value(phi_u);
val_u(isnan(val_u)) = 0;



K = value(phi_u) / value(phi_x);

x_init = [0;0;0;0];
x = zeros(state_dim,T);

for i = 1:T-1
    feedback_control = zeros(input_dim,1);
    for j = 1:i
        % Feedback gain exists for all previous timesteps (causal system)
        feedback_control = feedback_control + K((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim)*(x(:,j) - unstacked_z(:,j));
    end
    noise = randn(4, 1);     % random vector from N(0,1)
    noise = noise / norm(noise);     % normalize to have norm 1
    x(:,i+1) = A_t*x(:,i) + B_t*(feedback_control + unstacked_v(:,i)) + noise;
end
plot(x(1,:),x(2,:),"b","DisplayName","Disturbed w/ Feedback Control");

