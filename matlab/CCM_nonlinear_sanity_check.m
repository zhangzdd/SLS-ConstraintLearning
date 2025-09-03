% YALMIP-based Control Contraction Metric (CCM) synthesis
clear; clc;close all;
yalmip('clear');

% System dynamics
% Double integrator dynamics
Ts = 1;
A = [0 0 1 0;0 0 0 1; 0 0 0 0; 0 0 0 0]*Ts;
% B_t = [1/2*Ts^2 0;0 1/2*Ts^2;Ts 0;0 Ts]; 
% Omit the second order terms
B = [0 0;0 0;1 0;0 1]*Ts;
n = size(A, 1);

% Decision variables
W = sdpvar(n, n, 'symmetric');  % W = M^{-1}
rho = sdpvar(1);                % scalar multiplier
lambda = 0.5;                   % contraction rate

% LMI condition (from Finsler-transformed CCM inequality)
LMI = A*W + W*A' - rho*(B*B') + 2*lambda*W;

% Constraints
delta_x = sdpvar(4,1);

% constraints = [W >= 1e-2*eye(n), rho >= 0.2, LMI <= -1e-4*eye(n)];
% Solve via sos
constraints = [W >= 1e-2*eye(n), rho >= 0.2, sos(-delta_x'*LMI*delta_x)];

% constraints = [W <= 1e-2*eye(n), rho <= 1e-3];
% Solve
options = sdpsettings('solver', 'mosek', 'verbose', 2);
sol = optimize(constraints, [], options,[W(:);rho]);

% Check and display
if sol.problem == 0
    Wsol = value(W);
    rhosol = value(rho);
    M = inv(Wsol);

    % Compute feedback gain
    K = -(1/2*rhosol) * B' * M;  % 1×2 gain
    disp('Feasible CCM found:');
    disp('W ='); disp(Wsol);
    disp('M = inv(W) ='); disp(inv(Wsol));
    disp('K = ');disp(K);
else
    disp('No feasible solution found.');
    disp(sol.info);
end