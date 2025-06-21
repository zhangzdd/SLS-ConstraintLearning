clear;clc;close all;
Theta_MAX = 20;


% 3 box 4 boundary
N_m = 1;
dimAffine = 4;
theta = sdpvar(N_m,4,'full');
xu = theta(1);
xl = theta(2);
yu = theta(3);
yl = theta(4);
theta_l = reshape(theta,[],1);


elem_inv=0;


for i =1:N_m
    b = binvar(6,1);
    elem_inv = elem_inv + b(end)*( 1-xu(i)-xl(i) ) ...
                + b(end-1)*( 1-yu(i)-yl(i) ) ...
                + b(end-2)*( abs(xu(i))-Theta_MAX ) ...
                + b(end-3)*( abs(xl(i))-Theta_MAX ) ...
                + b(end-4)*( abs(yu(i))-Theta_MAX ) ...
                + b(end-5)*( abs(yl(i))-Theta_MAX );
end

tic
% query point
theta_q = [5;15;15;4];

constraints_inv = [];
constraints_inv = [constraints_inv, b(end)*( 1-xu(i)-xl(i) ) ...
                + b(end-1)*( 1-yu(i)-yl(i) ) ...
                + b(end-2)*( abs(xu(i))-Theta_MAX ) ...
                + b(end-3)*( abs(xl(i))-Theta_MAX ) ...
                + b(end-4)*( abs(yu(i))-Theta_MAX ) ...
                + b(end-5)*( abs(yl(i))-Theta_MAX ) >= 1e-2];

eff_idx = [2,3];


ops = sdpsettings('solver','gurobi','verbose', 2);
optimize(constraints_inv,max(abs(theta_l(eff_idx) - theta_q(eff_idx))), ops);
safe_box_size = value(max(abs(theta_l(eff_idx) - theta_q(eff_idx))));

disp('query point: ');
disp(theta_q);
disp('closest point: ');
disp(value(theta_l));
disp('distance');
disp(safe_box_size);

value(b(end)*( 1-xu(i)-xl(i) ) ...
                + b(end-1)*( 1-yu(i)-yl(i) ) ...
                + b(end-2)*( abs(xu(i))-Theta_MAX ) ...
                + b(end-3)*( abs(xl(i))-Theta_MAX ) ...
                + b(end-4)*( abs(yu(i))-Theta_MAX ) ...
                + b(end-5)*( abs(yl(i))-Theta_MAX ))
disp('Finish Optimization')
toc