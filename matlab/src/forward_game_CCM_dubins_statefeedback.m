% Improved 4D nonholonomic car CCM with systematic debugging
clear; clc; yalmip('clear'); close all;

%% --- Step 1: Analyze polynomial approximation quality ---
phiL = pi/8; phiU = pi - pi/8;  % START with smaller domain
degFit = 5;  % Increase degree for better approximation
Nfit = 1000; 
phigrid = linspace(phiL, phiU, Nfit);

% Fit polynomials
c_cf = polyfit(phigrid, cos(phigrid), degFit);
s_cf = polyfit(phigrid, sin(phigrid), degFit);

% Check approximation quality
cos_approx = polyval(c_cf, phigrid);
sin_approx = polyval(s_cf, phigrid);
cos_error = max(abs(cos(phigrid) - cos_approx));
sin_error = max(abs(sin(phigrid) - sin_approx));

fprintf('Polynomial approximation analysis:\n');
fprintf('Max cos error: %.2e\n', cos_error);
fprintf('Max sin error: %.2e\n', sin_error);
fprintf('Domain: [%.2f, %.2f] radians\n', phiL, phiU);

% Plot approximation quality
figure(1); clf;
subplot(2,1,1);
plot(phigrid, cos(phigrid), 'b-', 'LineWidth', 2); hold on;
plot(phigrid, cos_approx, 'r--', 'LineWidth', 1.5);
legend('cos(\phi)', sprintf('poly approx (deg %d)', degFit));
title('Cosine Approximation Quality');
grid on;

subplot(2,1,2);
plot(phigrid, sin(phigrid), 'b-', 'LineWidth', 2); hold on;
plot(phigrid, sin_approx, 'r--', 'LineWidth', 1.5);
legend('sin(\phi)', sprintf('poly approx (deg %d)', degFit));
title('Sine Approximation Quality');
grid on;

%% --- Step 2: Set up symbolic variables ---
phi = sdpvar(1,1);
v = sdpvar(1,1);
cos_p = 0; sin_p = 0;
for k = 0:degFit
    cos_p = cos_p + c_cf(k+1)*phi^(degFit-k);
    sin_p = sin_p + s_cf(k+1)*phi^(degFit-k);
end

jac_cos_p = jacobian(cos_p,phi);
jac_sin_p = jacobian(sin_p,phi);

%% --- Step 3: System dynamics ---

% % A is the variation delta_f/delta_dx
% f = [vcos(phi),vsin(phi),0,0]', x = [x,y,phi,v]
A = [ 0 0  v*jac_cos_p  cos_p;
      0 0  v*jac_sin_p  sin_p;
      0 0   0           0   ;
      0 0   0           0   ];

% % Sanity check: linear dynamics working
% A = [ 0 0 1 0;
%       0 0 0 1;
%       0 0 0 0;
%       0 0 0 0];

B = [0 0; 0 0; 1 0; 0 1];


% Decision variables
W = sdpvar(4, 4, 'symmetric');  % W = M^{-1}
rho = sdpvar(1);                % scalar multiplier
slack = sdpvar(1);
lambda = 0.5;                   % contraction rate

% LMI condition (from Finsler-transformed CCM inequality)
LMI = A*W + W*A' - rho*(B*B') + 2*lambda*W;

% Constraints
delta_x = sdpvar(4,1);

% %% --- Step 4: Domain-restricted SOS (scope on phi and v) ---
% % Box (scope) for phi and v
% vL = 0.5;          % <-- adjust as needed
% vU = 1;          % <-- adjust as needed
% 
% phi_box = (phi - phiL)*(phiU - phi);   % >= 0 on [phiL, phiU]
% v_box   = (v   - vL )*(vU   - v  );    % >= 0 on [vL, vU]
% 
% % The polynomial we want nonnegative on the box:
% p = -delta_x.'*LMI*delta_x;            % should be >= 0 on the scope
% 
% % SOS multipliers s1(phi,v), s2(phi,v) of chosen degree
% degMult = 2;                           % small degree often suffices
% [s1,cS1] = polynomial([phi v], degMult);
% [s2,cS2] = polynomial([phi v], degMult);
% 
% % Enforce: p - s1*phi_box - s2*v_box is SOS on R^2, and s1,s2 are SOS
% sos_constr = [ sos(p - s1*phi_box - s2*v_box), sos(s1), sos(s2) ];
% 
% % Replace your previous global-SOS + slack with domain-restricted SOS
% 
% constraints = [50 * eye(4) >= W >= 1e-2*eye(4), rho >= 0.2, sos_constr];
% 
% % (Optional) If you want a slightly tighter certificate, add a mixed term:
% % [s3,cS3] = polynomial([phi v], degMult);
% % constraints = [constraints, sos(s3)];
% % constraints = [constraints, sos(p - s1*phi_box - s2*v_box - s3*phi_box*v_box)];
% 
% % Solve (you can include polynomial coeffs so the solver knows all dec vars)
% decvars = [W(:); rho; cS1; cS2]; % add cS3 if you used s3
% options = sdpsettings('solver','mosek','verbose',2);
% sol = solvesos(constraints, [], options, decvars);  % solvesos recommended for SOS
% 
% % Check and display
% if sol.problem == 0
%     Wsol = value(W);
%     rhosol = value(rho);
%     M = inv(Wsol);
% 
%     % Compute feedback gain
%     K = -(1/2*rhosol) * B' * M;  % 1×2 gain
%     disp('Feasible CCM found:');
%     disp('W ='); disp(Wsol);
%     disp('M = inv(W) ='); disp(inv(Wsol));
%     disp('K = ');disp(K);
% else
%     disp('No feasible solution found.');
%     disp(sol.info);
% end
% 
%% --- Use exact trig via (c,s) variables; matrix-SOS on a compact set ---

% New polynomial variables for trig (instead of polynomial fits)
c = sdpvar(1,1);   % cos(phi)
s = sdpvar(1,1);   % sin(phi)

% Dynamics Jacobian using (c,s)
A = [ 0 0  v*(-s)  c;
      0 0  v*( c)  s;
      0 0   0      0;
      0 0   0      0 ];

% LMI (Finsler CCM form)
LMI = A*W + W*A' - rho*(B*B') + 2*lambda*W;
LMI = 0.5*(LMI + LMI');      % symmetrize

% --- Domain as polynomials ---
vL = 0.5; vU = 1.0;
cL = 0;   % = 0
cU = 1;   % = sqrt(3)/2
sL = 0;   % = 1/2
sU = 1;   % = 1

g_v = (v - vL)*(vU - v);     % >=0 on v-box
g_c = (c - cL)*(cU - c);     % >=0 on c-interval
g_s = (s - sL)*(sU - s);     % >=0 on s-interval
eq_circle = c^2 + s^2 - 1;   % =0 on unit circle

% Bounds on W for conditioning & margin gamma
alpha = 1e-2; beta = 50;
gamma = sdpvar(1,1);
gamma_min = 1e-3;

Wbox = [ W >= alpha*eye(4), W <= beta*eye(4) ];

% SOS multipliers (box) and free equality multiplier (DO NOT constrain h to be SOS)
degMult = 2;   % try 2; if infeasible, bump to 4
[s_v, cs_v] = polynomial([v c s], degMult);
[s_c, cs_c] = polynomial([v c s], degMult);
[s_s, cs_s] = polynomial([v c s], degMult);
[  h,   ch] = polynomial([v c s], 1);        % low-degree free multiplier for eq_circle

% Matrix-SOS certificate:
%  -(LMI + gamma*I) - s_v*g_v*I - s_c*g_c*I - s_s*g_s*I - h*(c^2+s^2-1)*I  is SOS-matrix
constrSOS = [ sos(s_v), sos(s_c), sos(s_s), ...
              sos( -(LMI + gamma*eye(4)) ...
                    - s_v*g_v*eye(4) - s_c*g_c*eye(4) - s_s*g_s*eye(4) ...
                    - h*eq_circle*eye(4) ) ];

constraints = [ Wbox, rho >= 0, gamma >= gamma_min, constrSOS ];

% Mild regularizer; keep rho small to avoid huge gains later
obj = 1e-3*trace(W) + rho + 1e-1*gamma;

decvars = [W(:); rho; gamma; cs_v; cs_c; cs_s; ch];

options = sdpsettings('solver','mosek','verbose',2, ...
                      'sos.model',2,'sos.scale','on');
sol = solvesos(constraints, obj, options, decvars);

% Readout as before
if sol.problem==0
    Wsol = value(W); rhosol = value(rho);
    M = inv(Wsol);
    K = -(rhosol/2) * B' * M;
    disp('Feasible CCM (SOS) found');
    disp('rho ='); disp(rhosol); disp('W='); disp(Wsol); disp('K='); disp(K);
else
    disp('No feasible solution (SOS).'); disp(sol.info);
end
%% Formulate the forward optimization with LTV system premise
import casadi.*
opti = casadi.Opti();

T = 20; % Time horizon
Ts = 0.2; % Sampling time
state_dim = 4;
input_dim = 2;
disturbance_level = 0.05; % The magnitude of disturbances
% mu = 
% True dynamics x_dot = f(x) + g(x)*u
f = @(x) Ts*[x(4)*cos(x(3)),x(4)*sin(x(3)),0,0]';


% 4D car dynamics (linearized about (phi_nom, v_nom), then Euler discretized)
% x = [x; y; phi; v],  u = [u_phi; u_v]
% Linearized dubins car dynamics, state dependent

% Nominal trajectory z,v
z = opti.variable(state_dim*T,1);
v = opti.variable(input_dim*T,1);

A_c = @(phi_nom,v_nom) [ 0  0  -v_nom*sin(phi_nom)   cos(phi_nom) ;
        0  0   v_nom*cos(phi_nom)   sin(phi_nom) ;
        0  0    0                   0 ;
        0  0    0                   0 ];
B_c = [ 0  0 ;
        0  0 ;
        1  0 ;
        0  1 ];

B_t = Ts*B_c;


% Populate the block matrices with desicion variables
num_repetitions_A = T - 1;
num_repetitions_B = T - 1;
num_repetitions_K = T - 1;
matrices_A = cell(1, num_repetitions_A);
for i = 1:num_repetitions_A+1
    if i == T
        matrices_A{i} = zeros(state_dim);
    else
        matrices_A{i} = A_c(z((i-1)*state_dim+3),z((i-1)*state_dim+4))*Ts+eye(state_dim);
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
matrices_K = cell(1, num_repetitions_K);
for i = 1:num_repetitions_K+1
    if i == T
        matrices_K{i} = zeros(size(K));
    else
        matrices_K{i} = K;
    end
end
A = blkdiag(matrices_A{:});
B = blkdiag(matrices_B{:});
K = blkdiag(matrices_K{:});

Z = createBlockDownshiftOperator(state_dim,T);

%% Calculate the Phi matrices from K (this translates to constraints on phi matrices in the nonlinear case)

% Construct the phi block matrices
phi_x = opti.variable(T*state_dim, T*state_dim,'full');
phi_u = opti.variable(T*input_dim, T*state_dim,'full');

constraints = [];

% Populate the phi_x and phi_u as blcok lower triangular matrices
for i = 1:T
    for j = 1:T
        if j>i
            % opti.subject_to(phi_x((i-1)*state_dim+1:i*state_dim,(j-1)*state_dim+1:j*state_dim) == zeros(state_dim,state_dim));
            % opti.subject_to(phi_u((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) == zeros(input_dim,state_dim));
            % phi_x((i-1)*state_dim+1:i*state_dim,(j-1)*state_dim+1:j*state_dim) = zeros(state_dim,state_dim);
            phi_u((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim) = zeros(input_dim,state_dim);
        end
    end
end


C = [eye(T*state_dim) - Z*A, -Z*B;
            K              , -eye(input_dim*T)];

opti.subject_to(C*[phi_x;phi_u] == [eye(state_dim*T);zeros(input_dim*T,state_dim*T)]);

% Try SLS without K
% opti.subject_to([eye(T*state_dim) - Z*A, -Z*B]*[phi_x;phi_u] == eye(state_dim*T));
% learned_phi = pinv(C)*[eye(state_dim*T);zeros(input_dim*T,state_dim*T)];
% phi_x = learned_phi(1:T*state_dim,:);
% phi_u = learned_phi(T*state_dim+1:end,:);
% 
% disp("nomr phi_x");disp(norm(phi_x,'fro'));
% disp("nomr phi_u");disp(norm(phi_u,'fro'));
% disp("norm K");disp(norm(K,'fro'));


%% Populate some placeholder nominal trajectory
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


constrained_states_upper_limited = 4;
tube_size = cell(T,1);
for dim = 1:DimAffine
    for k = 1:T-1
        phi_kx = phi_x((k-1)*state_dim+1:k*state_dim,:);
        phi_ku = phi_u((k-1)*input_dim+1:k*input_dim,:);
        z_k = z((k-1)*state_dim+1:k*state_dim);
        v_k = v((k-1)*input_dim+1:k*input_dim);
        z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
        opti.subject_to(z_k_plus_1 == z_k + Ts * [ z_k(4)*cos(z_k(3));
                                           z_k(4)*sin(z_k(3));
                                           v_k(1);
                                           v_k(2) ]);       
        % constraints = [constraints, A_poly(dim,:)*z_k(1:constrained_states_upper_limited) + norm(A_poly(dim,:)*phi_kx(1:constrained_states_upper_limited,:),1) <= b_poly(dim)];
        robust_affine_constraints = 0;
        for j = 1:k
            % robust_affine_constraints = robust_affine_constraints + disturbance_level * norm(A_poly(dim,:)*phi_kx(:,(j-1)*state_dim+1:j*state_dim),1);
            rj = A_poly(dim,:) * phi_kx(:, (j-1)*state_dim+1:j*state_dim); % row → vector
            t = opti.variable(1,length(rj));  % auxiliary nonnegative variables
            
            opti.subject_to(t >=  rj);
            opti.subject_to(t >= -rj);
            
            robust_affine_constraints = robust_affine_constraints + disturbance_level * sum2(t);
        end
        
        opti.subject_to(A_poly(dim,:)*z_k + robust_affine_constraints <= b_poly(dim));
        tube_size{k} = robust_affine_constraints;
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
opti.subject_to(z_init(1:state_dim) == [0;0;0;0]);
opti.subject_to(z_terminal(1:state_dim) == [0.;10;0;0]);
% constraints = [constraints, v_terminal(1:input_dim) == [0;0]];


% Terminal robust constraint
for dim = 1:DimAffine
    robust_affine_constraints = 0;
    for j = 1:T
        % robust_affine_constraints = robust_affine_constraints + disturbance_level * norm(A_poly(dim,:)*phi_x((T-1)*state_dim+1:T*state_dim,(j-1)*state_dim+1:j*state_dim),1);
        rj = A_poly(dim,:) * phi_kx(:, (j-1)*state_dim+1:j*state_dim); % row → vector
        t = opti.variable(1,length(rj));  % auxiliary nonnegative variables
        
        opti.subject_to(t >=  rj);
        opti.subject_to(t >= -rj);
        
        robust_affine_constraints = robust_affine_constraints + disturbance_level * sum2(t);
    end
    opti.subject_to(A_poly(dim,:)*z_terminal + robust_affine_constraints <= b_poly(dim));
end


% SLS constraints
% % constraints = [constraints, [eye(T*state_dim) - Z*A, -Z*B]*[phi_x;phi_u] == eye(T*state_dim)];


% Objective
objective = 0;
for k = 1:T-1
    z_k = z((k-1)*state_dim+1:k*state_dim);
    z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
    
    % The objective is two-fold: trying to maximize x-coordinates, and
    % trying to smooth the overall trajectory
    objective = objective + sum((z_k_plus_1(1:2) - z_k(1:2)).^2) - z_k(1);
end

% Explicit constraints on input level
% opti.subject_to(v <= 10);
% opti.subject_to(v >= -10);
% opti.subject_to(v(:,end) == 0);

% Penalize the phi_u
% objective = objective + norm(phi_u, 'fro');



% Solve
opti.minimize(objective);
opti.solver('ipopt');
sol = opti.solve();

%%
% Plot nominal trajectory (z)
figure;hold on;
unstacked_z = reshape(sol.value(z),[state_dim, T]);
unstacked_v = reshape(sol.value(v),[input_dim, T]);
plot(unstacked_z(1,:),unstacked_z(2,:),"g","DisplayName","Nominal");
xlim([-10,10]);ylim([-10,10]);

%% Roll out the trajectory with feedback matrix K

num_rollout = 100;
error_signal_feedback = cell(num_rollout,1);
error_signal_openloop = cell(num_rollout,1);
state_trajectory_closedloop = cell(num_rollout,1);
input_trajectory_closedloop = cell(num_rollout,1);

f = @(x) Ts*[x(4)*cos(x(3)),x(4)*sin(x(3)),0,0]';





for rollout_cnt = 1:num_rollout
    % Roll out trajectory with noise to the dynamics, WITHOUT the feedback
    x_init = [0;0;0;0]; % same as nominal trajectory initial state
    x = zeros(state_dim,T);
    
    for i = 1:T-1
        % noise = randn(4, 1);     % random vector from N(0,1)
        noise = disturbance_level * (rand(4,1)*2 - 1);
        new_x = x(:,i) + f(x(:,i)) + B_t*unstacked_v(:,i) + noise;
        x(:,i+1) = new_x;
    end
    plot(x(1,:),x(2,:),"r","DisplayName","Disturbed");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"r","DisplayName","Error signal without Feedback Control")
    error_signal_openloop{rollout_cnt} = [x(1,1:end) - unstacked_z(1,1:end);x(2,1:end) - unstacked_z(2,1:end)];
    
    % Roll out trajectory with noise to the dynamics, WITH the feedback
    
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
            x(:,i+1) = x(:,i) + f(x(:,i)) + B_t*(feedback_control + unstacked_v(:,i)) + noise;
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
% save('forward_game_data.mat','state_trajectory_closedloop','input_trajectory_closedloop','unstacked_z','unstacked_v','state_dim','input_dim','DimAffine','T','num_rollout','val_phi_u','val_phi_x','Z','A','B','A_t','B_t', ...
%     'disturbance_level');
xlabel('x',Interpreter='latex');
ylabel('y',Interpreter='latex');

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
axis equal;xlim([-5,15]);ylim([0,16]);


% 2) Shade the forbidden half-space (to the right of the constraint line)
yl = ylim; xr = xlim;
hForbid = patch([b_poly(1) xr(2) xr(2) b_poly(1)], [yl(1) yl(1) yl(2) yl(2)], ...
    [0 0 0], 'FaceAlpha',0.06, 'EdgeColor','none', 'HandleVisibility','off');
uistack(hForbid,'bottom');  % keep it behind the trajectories
text(b_poly(1)+4, mean(yl), '\textbf{unsafe}', 'Interpreter','latex', ...
     'Rotation',45, 'HorizontalAlignment','left', 'Color',[0 0 0],'FontSize',30);




