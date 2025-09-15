%%
% This code tests the minimum number of rollouts we need to ensure exact
% recovery of K
%% Run roll outs at multiple times with noise signal
num_rollout = 10000;
error_signal_feedback = cell(num_rollout,1);
error_signal_openloop = cell(num_rollout,1);
state_trajectory_closedloop = cell(num_rollout,1);
input_trajectory_closedloop = cell(num_rollout,1);

% dynamics jacobians
A_t = zeros(12,12);
A_t(1,7)  = 1;        % xdot = vx
A_t(2,8)  = 1;        % ydot = vy
A_t(3,9)  = 1;        % zdot = vz
A_t(4,10) = 1;        % phidot = p
A_t(5,11) = 1;        % thetadot = q
A_t(6,12) = 1;        % psidot = r
A_t(7,5)  =  g;       % vxdot ≈  g*theta
A_t(8,4)  = -g;       % vydot ≈ -g*phi
% remaining entries are zero

B_t = zeros(12,4);
B_t(10,1) = 1/Ix;     % pdot = tau_x/Ix
B_t(11,2) = 1/Iy;     % qdot = tau_y/Iy
B_t(12,3) = 1/Iz;     % rdot = tau_z/Iz
B_t(9,4)  = 1/m;      % vzdot = T/m - g  (the -g is in A via equilibrium)



for rollout_cnt = 1:num_rollout
    % Roll out trajectory with noise to the dynamics, WITHOUT the feedback
    x_init = [0;0;0;0]; % same as nominal trajectory initial state
    x = zeros(state_dim,T);
    
    for i = 1:T-1
        % noise = randn(4, 1);     % random vector from N(0,1)
        noise = disturbance_level * (rand(state_dim,1)*2 - 1);
        new_x = A_t*x(:,i) + B_t*unstacked_v(:,i) + noise;
        x(:,i+1) = new_x;
    end
    % plot(x(1,:),x(2,:),"r","DisplayName","Disturbed");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"r","DisplayName","Error signal without Feedback Control")
    error_signal_openloop{rollout_cnt} = [x(1,1:end) - unstacked_z(1,1:end);x(2,1:end) - unstacked_z(2,1:end)];
    
    % Roll out trajectory with noise to the dynamics, WITH the feedback
    
    x_init = disturbance_level * (rand(state_dim,1)*2 - 1);
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
        noise = disturbance_level * (rand(state_dim,1)*2 - 1);
        if i<=T-1
            x(:,i+1) = A_t*x(:,i) + B_t*(feedback_control + unstacked_v(:,i)) + noise;
        end
        feedback_u(:,i) = feedback_control;
    end
    % plot(x(1,1:end),x(2,1:end),"b","DisplayName","Disturbed w/ Feedback Control");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"b","DisplayName","Error signal w/ Feedback Control")
    error_signal_feedback{rollout_cnt} = [x(1,1:end) - unstacked_z(1,1:end);x(2,1:end) - unstacked_z(2,1:end)];
    state_trajectory_closedloop{rollout_cnt} = x;
    input_trajectory_closedloop{rollout_cnt} = feedback_u;
end


for min_rollout_cnt = 1:num_rollout
    X = zeros(state_dim*T,min_rollout_cnt-1);
    U = zeros(input_dim*T,min_rollout_cnt-1);
    for t = 1:T
        x_diff = {};
        u_diff = {};
        x_diff_unstacked = zeros(state_dim,min_rollout_cnt-1);
        u_diff_unstacked = zeros(input_dim,min_rollout_cnt-1);
        for rollout_cnt = 1:min_rollout_cnt-1
            x_diff{rollout_cnt} = state_trajectory_closedloop{rollout_cnt+1}(:,t) - state_trajectory_closedloop{rollout_cnt}(:,t);
            u_diff{rollout_cnt} = input_trajectory_closedloop{rollout_cnt+1}(:,t) - input_trajectory_closedloop{rollout_cnt}(:,t);
            x_diff_unstacked(:,rollout_cnt) = x_diff{rollout_cnt};
            u_diff_unstacked(:,rollout_cnt) = u_diff{rollout_cnt};
        end
        X((t-1)*state_dim+1:t*state_dim,:) = x_diff_unstacked;
        U((t-1)*input_dim+1:t*input_dim,:) = u_diff_unstacked;
    end
    
    % This is supposed to a lower triangular matrix
    K = U * pinv(X);
    % Compare against the ground truth
    % is_close = norm(K - val_phi_u / val_phi_x, 'fro') < 1e-5
    isFullRank = rank(K) == min(size(K));
    if isFullRank
        break
    end
    disp(min_rollout_cnt);
end

disp(min_rollout_cnt);