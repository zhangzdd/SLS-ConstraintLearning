%% Formulate the forward optimization with LTV system premise
clear;clc;
import casadi.*
opti = casadi.Opti();

T  = 15;           % Time horizon
Ts = 0.2;          % Sampling time
state_dim = 6;     % CHANGED: [x; z; theta; vx; vz; q]
input_dim = 2;     % [u1_tilde; u2]
disturbance_level = 0.05;

% --- Quadrotor constants (hover linearization reference)
m  = 1.0;          % mass [kg]
g  = 9.81;         % gravity [m/s^2]
Iy = 0.02;         % pitch inertia [kg m^2]

% Nominal trajectory variables
z = opti.variable(state_dim*T,1);    % stacked states
v = opti.variable(input_dim*T,1);    % stacked inputs [u1_tilde; u2]

% ===== Trajectory-dependent linearization (continuous-time Jacobians) =====
% Nonlinear CT dynamics (for reference):
% xdot = [ vx ;
%          vz ;
%          q  ;
%         -(u1/m)*sin(theta) ;
%          (u1/m)*cos(theta) - g ;
%          (1/Iy)*u2 ],  where  u1 = mg + u1_tilde
%
% A_c(theta, u1): d f / d x
A_c = @(theta_nom, u1_nom) [ ...
    0   0   0                  1   0   0 ;
    0   0   0                  0   1   0 ;
    0   0   0                  0   0   1 ;
    0   0  -(u1_nom/m)*cos(theta_nom)  0   0   0 ;
    0   0  -(u1_nom/m)*sin(theta_nom)  0   0   0 ;
    0   0   0                  0   0   0 ];

% B_c(theta): d f / d u_tilde  (inputs are [u1_tilde; u2])
B_c = @(theta_nom) [ ...
    0       0 ;
    0       0 ;
    0       0 ;
   -(1/m)*sin(theta_nom)   0 ;
    (1/m)*cos(theta_nom)   0 ;
    0                 1/Iy ];

% ===== Build block-diagonal A,B along the nominal trajectory =====
num_repetitions_A = T - 1;
num_repetitions_B = T - 1;
num_repetitions_K = T - 1;

matrices_A = cell(1, num_repetitions_A);
matrices_B = cell(1, num_repetitions_B);

for i = 1:num_repetitions_A+1
    if i == T
        matrices_A{i} = zeros(state_dim);
    else
        % Extract nominal (theta_k, u1tilde_k) at step k=i
        theta_k    = z((i-1)*state_dim + 3);
        u1tilde_k  = v((i-1)*input_dim + 1);
        u1_k       = m*g + u1tilde_k;                  % total thrust at nominal
        A_k_ct     = A_c(theta_k, u1_k);
        matrices_A{i} = eye(state_dim) + Ts * A_k_ct;  % Euler discretization
    end
end

for i = 1:num_repetitions_B+1
    if i == T
        matrices_B{i} = zeros(state_dim, input_dim);
    else
        theta_k   = z((i-1)*state_dim + 3);
        B_k_ct    = B_c(theta_k);
        matrices_B{i} = Ts * B_k_ct;                   % Euler discretization
    end
end

% K blocks
% --- PD gains
omega_z = 1.0; zeta_z = 0.9;
k_pz = m*omega_z^2;          % = 1.0
k_dz = 2*m*zeta_z*omega_z;   % = 1.8

omega_x = 0.6; zeta_x = 0.9;
k_px = omega_x^2;            % = 0.36
k_dx = 2*zeta_x*omega_x;     % = 1.08

omega_t = 5.0; zeta_t = 0.9;
k_pt = omega_t^2;            % = 25
k_dt = 2*zeta_t*omega_t;     % = 9

% --- Linear (memoryless) K mapping: u = K_pd * e,
% e = [ex ez etheta evx evz eq]
K_pd = [ 0            -k_pz   0                0             -k_dz   0 ;
        -Iy*(k_pt*k_px/g)  0   -Iy*k_pt   -Iy*(k_pt*k_dx/g)   0    -Iy*k_dt ];

% Use this as your per-step block (keeps your structure unchanged)
K_block = K_pd;   % <--- replace the previous zero placeholder

matrices_K = cell(1, num_repetitions_K);
for i = 1:num_repetitions_K+1
    if i == T
        matrices_K{i} = zeros(size(K_block));
    else
        matrices_K{i} = K_block;
    end
end

A = blkdiag(matrices_A{:});
B = blkdiag(matrices_B{:});
K = blkdiag(matrices_K{:});

Z = createBlockDownshiftOperator(state_dim,T);

%% SLS response variables (unchanged)
phi_x = opti.variable(T*state_dim, T*state_dim,'full');
phi_u = opti.variable(T*input_dim, T*state_dim,'full');

% Keep lower-triangular hint for phi_u (as in your code)
for i = 1:T
    for j = 1:T
        if j > i
            phi_u((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) = zeros(input_dim,state_dim);
        end
    end
end

C = [eye(T*state_dim) - Z*A, -Z*B;
            K              , -eye(input_dim*T)];
opti.subject_to(C*[phi_x;phi_u] == [eye(state_dim*T);zeros(input_dim*T,state_dim*T)]);

%% Tube constraints etc. (unchanged structure; resized to 6 states)
A_poly = [ 1  0  0  0  0  0 ;   % +x <= b1
          -1  0  0  0  0  0 ;   % -x <= b2
           0  1  0  0  0  0 ;   % +z <= b3
           0 -1  0  0  0  0 ];  % -z <= b4
b_poly = [3; 5; 2; 2];
DimAffine = 1;   % use first row only, like your code

constrained_states_upper_limited = state_dim;
tube_size = opti.variable(1,T);
%% ===== Nominal dynamics constraint: nonlinear quadrotor =====
for dim = 1:DimAffine
    for k = 1:T-1
        phi_kx = phi_x((k-1)*state_dim+1:k*state_dim,:);
        % phi_ku = phi_u((k-1)*input_dim+1:k*input_dim,:);  % (unused here)
        z_k    = z((k-1)*state_dim+1:k*state_dim);
        v_k    = v((k-1)*input_dim+1:k*input_dim);
        z_kp1  = z(k*state_dim+1:(k+1)*state_dim);

        theta_k = z_k(3);
        u1_k    = m*g + v_k(1);
        u2_k    = v_k(2);

        % Nonlinear Euler step: z_{k+1} = z_k + Ts * f(z_k, v_k)
        opti.subject_to(z_kp1 == z_k + Ts * [ ...
            z_k(4);                             % vx
            z_k(5);                             % vz
            z_k(6);                             % q
           -(u1_k/m)*sin(theta_k);              % vx_dot
            (u1_k/m)*cos(theta_k) - g;          % vz_dot
            (1/Iy)*u2_k ]);                     % q_dot

        % Robust affine tightening (unchanged)
        robust_affine_constraints = 0;
        for j = 1:k
            rj = A_poly(dim,:) * phi_kx(:, (j-1)*state_dim+1:j*state_dim); % row → vector
            t  = opti.variable(1,length(rj));  % auxiliary nonnegative variables
            opti.subject_to(t >=  rj);
            opti.subject_to(t >= -rj);
            robust_affine_constraints = robust_affine_constraints + disturbance_level * sum2(t);
        end

        opti.subject_to(A_poly(dim,:)*z_k + robust_affine_constraints <= b_poly(dim));
        opti.subject_to(tube_size(k) == robust_affine_constraints);
    end
end

% Start and terminal constraints
z_init     = z(1:state_dim);
z_terminal = z((T-1)*state_dim+1:T*state_dim);
opti.subject_to(z_init == [0;0;0;0;0;0]);
% opti.subject_to(z_terminal == [0;10;0;0;0;0]);
opti.subject_to(z_terminal(1:2) == [0;10]);
opti.subject_to(z_terminal(4:6) == [0;0;0]);

% Terminal robust constraint (fixed out-of-scope phi_kx reference)
for dim = 1:DimAffine
    robust_affine_constraints = 0;
    phi_kx_T = phi_x((T-1)*state_dim+1:T*state_dim,:); % terminal block
    for j = 1:T
        rj = A_poly(dim,:) * phi_kx_T(:, (j-1)*state_dim+1:j*state_dim);
        t  = opti.variable(1,length(rj));
        opti.subject_to(t >=  rj);
        opti.subject_to(t >= -rj);
        robust_affine_constraints = robust_affine_constraints + disturbance_level * sum2(t);
    end
    opti.subject_to(tube_size(T) == robust_affine_constraints);
    opti.subject_to(A_poly(dim,:)*z_terminal + robust_affine_constraints <= b_poly(dim));
end

%% Objective (unchanged)
objective = 0;
for k = 1:T-1
    z_k   = z((k-1)*state_dim+1:k*state_dim);
    z_kp1 = z(k*state_dim+1:(k+1)*state_dim);
    % Smooth planar motion (x,z) and forward progress in x
    objective = objective + sum((z_kp1(1:2) - z_k(1:2)).^2) - z_k(1);
end

opti.minimize(objective);
opti.solver('ipopt');
sol = opti.solve();

%% Plot nominal trajectory (x vs z)
figure; hold on;
unstacked_z = reshape(sol.value(z),[state_dim, T]);
unstacked_v = reshape(sol.value(v),[input_dim, T]);
plot(unstacked_z(1,:), unstacked_z(2,:), "g","DisplayName","Nominal");
xlim([-10,10]); ylim([-10,10]);

%% Roll out (open/closed loop) with FULL nonlinear model
num_rollout = 100;
error_signal_feedback = cell(num_rollout,1);
error_signal_openloop = cell(num_rollout,1);
state_trajectory_closedloop = cell(num_rollout,1);
input_trajectory_closedloop = cell(num_rollout,1);

% Full nonlinear continuous-time dynamics for rollout
f_ct = @(x,u) [ ...
    x(4);
    x(5);
    x(6);
   -((m*g + u(1))/m)*sin(x(3));
    ((m*g + u(1))/m)*cos(x(3)) - g;
    (1/Iy)*u(2) ];

for rollout_cnt = 1:num_rollout
    % --- Open-loop
    x = zeros(state_dim,T);
    for i = 1:T-1
        noise = disturbance_level * (2*rand(state_dim,1) - 1);
        ui    = unstacked_v(:,i);
        x(:,i+1) = x(:,i) + Ts * f_ct(x(:,i), ui) + noise;
    end
    % plot(x(1,:), x(2,:), "r","DisplayName","Disturbed (open-loop)");
    error_signal_openloop{rollout_cnt} = [x(1,:) - unstacked_z(1,:);
                                          x(2,:) - unstacked_z(2,:)];

    % --- Closed-loop with K (zeros unless you set K_block above)
    x = zeros(state_dim,T);
    x(:,1) = disturbance_level * (2*rand(state_dim,1) - 1);
    feedback_u = zeros(input_dim,T);

    for i = 1:T
        feedback_control = zeros(input_dim,1);
        for j = 1:i
            Kij = K((i-1)*input_dim+1:i*input_dim, (j-1)*state_dim+1:j*state_dim);
            feedback_control = feedback_control + Kij*(x(:,j) - unstacked_z(:,j));
        end
        noise = disturbance_level * (2*rand(state_dim,1) - 1);
        if i <= T-1
            ui = feedback_control + unstacked_v(:,i);
            x(:,i+1) = x(:,i) + Ts * f_ct(x(:,i), ui) + noise;
        end
        feedback_u(:,i) = feedback_control;
    end
    plot(x(1,:), x(2,:), "b","DisplayName","Disturbed w/ Feedback");
    error_signal_feedback{rollout_cnt} = [x(1,:) - unstacked_z(1,:);
                                          x(2,:) - unstacked_z(2,:)];
    state_trajectory_closedloop{rollout_cnt} = x;
    input_trajectory_closedloop{rollout_cnt} = feedback_u;
end
xline(b_poly(1),":",'LineWidth',2);
xlabel('x','Interpreter','latex'); ylabel('z','Interpreter','latex');
xlabel('x',Interpreter='latex');
ylabel('y',Interpreter='latex');

hNom = plot(unstacked_z(1,1:end),unstacked_z(2,1:end), "g", "DisplayName","Nominal Trajectory",'LineWidth',3);
lineLearn = xline(b_poly(1),"--",'LineWidth',2,'Color','k','DisplayName','Learned Constraint(s)');
lineTruth = xline(b_poly(1),'LineWidth',2,'Color','y','DisplayName','Grond Truth Constraint(s)');
hOpen = plot(nan, nan, 'r', 'DisplayName','Open-loop (disturbed)');
hFB   = plot(nan, nan, 'b', 'DisplayName','Feedback (disturbed)');
hStart = plot(unstacked_z(1,1), unstacked_z(2,1), 'o', 'LineStyle','none', ...
    'MarkerSize',8, 'MarkerFaceColor','g', 'MarkerEdgeColor','k', ...
    'DisplayName','Start');

hGoal  = plot(unstacked_z(1,end), unstacked_z(2,end), 'p', 'LineStyle','none', ...
    'MarkerSize',11, 'MarkerFaceColor','g', 'MarkerEdgeColor','k', ...
    'DisplayName','Goal');

legend([hNom hFB lineLearn lineTruth], 'Interpreter','latex', 'Location','best');



%% 
% 1) Axes cosmetics & LaTeX
ax = gca; box on; grid on; ax.Layer = 'top';
ax.TickDir = 'out'; ax.LineWidth = 1;
ax.FontName = 'Times New Roman'; ax.FontSize = 10;  % or your journal’s font
set(ax,'TickLabelInterpreter','latex');
axis equal;xlim([-5,15]);ylim([0,16]);


% 2) Shade the forbidden half-space (to the right of the constraint line)
yl = ylim; xr = xlim;
hForbid = patch([b_poly(1) xr(2) xr(2) b_poly(1)], [yl(1) yl(1) yl(2) yl(2)], ...
    [0 0 0], 'FaceAlpha',0.06, 'EdgeColor','none', 'HandleVisibility','off');
uistack(hForbid,'bottom');  % keep it behind the trajectories
text(b_poly(1)+4, mean(yl), '\textbf{unsafe}', 'Interpreter','latex', ...
     'Rotation',45, 'HorizontalAlignment','left', 'Color',[0 0 0],'FontSize',30);

%% Plot tubes seperately
figure(2);hold on
h_TB = plot(linspace(1,T,T),sol.value(tube_size) + unstacked_z(1,:),'Color','#D95319','DisplayName','Tube Bound');
h_cstrt = yline(b_poly(1),":",'LineWidth',2,'DisplayName','Constraint');
xlabel('Timestep','Interpreter','latex')
ylabel('x coordinate value','Interpreter','latex')
for rollout_cnt = 1:num_rollout
    plot(linspace(1,T,T),state_trajectory_closedloop{rollout_cnt}(1,:),'Color','b','DisplayName','x value in demonstration trajectories');
end
x_FB = plot(nan, nan,'Color','b','DisplayName','x component in demonstration trajectories');
legend([x_FB h_TB h_cstrt],'Interpreter','latex', 'Location','best')
xlim([1,T]);
