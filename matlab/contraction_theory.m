% CCM tracking with model error (fixed: continuous-time CCM, rho floor, smaller dt, saturation 
clear; clc; close all; yalmip('clear');

%% Continuous-time model for CCM synthesis
Ac = [0 0 1 0;
      0 0 0 1;
      0 0 0 0;
      0 0 0 0];
Bc = [0 0;
      0 0;
      1 0;
      0 1];

n = 4;
W   = sdpvar(n,n,'symmetric');
rho = sdpvar(1);
lambda = 0.5;

% Continuous-time CCM LMI
LMI = Ac*W + W*Ac' - rho*(Bc*Bc') + 2*lambda*W;

% Important: floor rho; and well-conditioned W
constraints = [W >= 1e-3*eye(n), rho >= 0.2, LMI <= -1e-4*eye(n)];
opts = sdpsettings('solver','sdpt3','verbose',0);
sol = optimize(constraints,[],opts);
assert(sol.problem==0, sol.info);

Wsol = value(W); M = inv(Wsol); rhosol = value(rho);
K = -(1/2*rhosol) * (Bc') * M;   % 2x4

%% Reference: slow circle
Ts_sim = 0.02;   % simulation step
N = 4000;        % 80 s
t = (0:N-1)*Ts_sim;

R = 6; 
omega = 2*pi/80;  % one rev in ~80 s -> gentle

px =  R*cos(omega*t);
py =  R*sin(omega*t);
vx = -R*omega*sin(omega*t);
vy =  R*omega*cos(omega*t);
ax = -R*omega^2*cos(omega*t);
ay = -R*omega^2*sin(omega*t);

x_ref = [px; py; vx; vy];
u_ref = [ax; ay];  % continuous-time accel

%% True plant with mismatch
alpha_u = 0.9;          % input gain error
c_d     = 0.05;         % viscous drag
b       = [0.03; -0.02];% constant accel bias
gust_mag = 0.1;         
gust_every = 250;       % every 5 s
u_max = 2.5;            % saturation

x_true = zeros(4,N);
x_true(:,1) = x_ref(:,1) + [1.5; -1; 0.4; -0.3];
u_hist = zeros(2,N-1);

rng(3);
for k=1:N-1
    e = x_true(:,k) - x_ref(:,k);
    u = u_ref(:,k) + K*e;                 % CCM tracker (ct metric -> static gain)
    % Saturation
    u = max(min(u, u_max), -u_max);
    u_hist(:,k) = u;

    gust = [0;0];
    if mod(k,gust_every)==0
        gust = gust_mag*(2*rand(2,1)-1);
    end

    % Continuous-time plant integrated with Euler
    p = x_true(1:2,k); v = x_true(3:4,k);
    a_true = alpha_u*u - c_d*v + b + gust;
    p_next = p + Ts_sim*v;
    v_next = v + Ts_sim*a_true;
    x_true(:,k+1) = [p_next; v_next];
end

%% Plots
figure('Color','w');
subplot(1,2,1); hold on; axis equal; grid on; box on;
plot(x_ref(1,:), x_ref(2,:),'k--','LineWidth',1.2);
plot(x_true(1,:),x_true(2,:),'LineWidth',2);
legend('reference','true','Location','best');
title('Position trajectory'); xlabel('x'); ylabel('y');

subplot(3,2,2); hold on; grid on; box on;
plot(t, vecnorm(x_true(1:2,:)-x_ref(1:2,:),2,1),'LineWidth',1.3);
ylabel('||p - p_{ref}||'); title('Position error');

subplot(3,2,4); hold on; grid on; box on;
plot(t, vecnorm(x_true(3:4,:)-x_ref(3:4,:),2,1),'LineWidth',1.3);
ylabel('||v - v_{ref}||'); title('Velocity error');

subplot(3,2,6); hold on; grid on; box on;
plot(t(1:end-1), vecnorm(u_hist,2,1),'LineWidth',1.2);
ylabel('||u||'); xlabel('t [s]'); title('Control effort (saturated)');

sgtitle(sprintf('CCM tracking (ct metric), rho=%.2f, u_{max}=%.1f', rhosol, u_max));
