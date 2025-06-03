
N = 3; %Time horizon
state_dim = 4;
input_dim = 2;

A_t = [0 0 1 0;0 0 0 1; 0 0 0 0; 0 0 0 0];
B_t = [0 0;0 0;1 0;0 1];


num_repetitions_A = N - 1;
num_repetitions_B = N - 1;
matrices_A = cell(1, num_repetitions_A);
for i = 1:num_repetitions_A+1
    if i == N
        matrices_A{i} = zeros(size(A_t));
    else
        matrices_A{i} = A_t;
    end
end
matrices_B = cell(1, num_repetitions_B);
for i = 1:num_repetitions_B+1
    if i == N
        matrices_B{i} = zeros(size(B_t));
    else
        matrices_B{i} = B_t;
    end
end
A = blkdiag(matrices_A{:});
B = blkdiag(matrices_B{:});

Z = createBlockDownshiftOperator(state_dim,N);

% Construct the phi block matrices
import casadi.*
opti = casadi.Opti();

phi_x = opti.variable(N*state_dim, N*state_dim);
phi_u = opti.variable(N*input_dim, N*state_dim);

% Populate the phi_x and phi_u as lower triangular matrices
for i = 1:N
    for j = 1:N
        if j>i
            opti.subject_to(phi_x((i-1)*state_dim+1:i*state_dim,(j-1)*state_dim+1:j*state_dim) == zeros(state_dim,state_dim));
            opti.subject_to(phi_u((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) == zeros(input_dim,state_dim));
        end
    end
end

% Tube constraint in the MPSF paper
A_x1 = [1 0 0 0];
A_x2 = [-1 0 0 0];
A_x3 = [0 1 0 0];
A_x4 = [0 -1 0 0];
b_x1 = 3;
b_x2 = -3;
b_x3 = 2;
b_x4 = -2;

% Nominal trajectory z,v
z = opti.variable(state_dim*N,1);
v = opti.variable(input_dim*N,1);


for k = 1:N-1
    phi_kx = phi_x((k-1)*state_dim+1:k*state_dim,:);
    % phi_ku = phi_u((k-1)*state_dim+1:k*state_dim,:);
    z_k = z((k-1)*state_dim+1:k*state_dim);
    v_k = v((k-1)*input_dim+1:k*input_dim);
    z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
    opti.subject_to(z_k_plus_1 == A_t*z_k + B_t*v_k);
    opti.subject_to(A_x1*z_k + norm(A_x1*phi_kx,1) <= b_x1);
end

% Start and terminal constraint
z_init = z((1-1)*state_dim+1:1*state_dim);
z_terminal = z((N-1)*state_dim+1:N*state_dim);
opti.subject_to(z_init(1:2) == [0;0]);
opti.subject_to(z_terminal(1:2) == [1;1]);

% SLS constraints
opti.subject_to([eye(N*state_dim) - Z*A, -Z*B]*[phi_x;phi_u] == eye(N*state_dim));


% Objective
objective = 0;
for k = 1:N-1
    z_k = z((k-1)*state_dim+1:k*state_dim);
    z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
    
    objective = objective + sum((z_k_plus_1 - z_k).^2);
end

opti.minimize(objective);
opti.solver('ipopt');
sol = opti.solve();


