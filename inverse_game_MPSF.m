% Time to implement the inverse game!
% First we need to invert K from the demonstrations
X = zeros(state_dim*T,num_rollout-1);
U = zeros(input_dim*T,num_rollout-1);
for t = 1:T
    x_diff = {};
    u_diff = {};
    x_diff_unstacked = zeros(state_dim,rollout_cnt-1);
    u_diff_unstacked = zeros(input_dim,rollout_cnt-1);
    for rollout_cnt = 1:num_rollout-1
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
is_close = norm(K - val_u / value(phi_x), 'fro') < 1e-5;
%% Roll out with learned K
figure(3); hold on;
for rollout_cnt = 1:num_rollout
    % Roll out trajectory with noise to the dynamics, WITHOUT the feedback
    x_init = [0;0;0;0]; % same as nominal trajectory initial state
    x = zeros(state_dim,T);
    
    for i = 1:T-1
        % noise = randn(4, 1);     % random vector from N(0,1)
        noise = disturbance_level * (rand(4,1)*2 - 1);
        new_x = A_t*x(:,i) + B_t*unstacked_v(:,i) + noise;
        x(:,i+1) = new_x;
    end
    plot(x(1,:),x(2,:),"r","DisplayName","Disturbed");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"r","DisplayName","Error signal without Feedback Control")    
    % Roll out trajectory with noise to the dynamics, WITH the feedback

    x_init = [0;0;0;0];
    x = zeros(state_dim,T);
    feedback_u = zeros(input_dim,T);
    
    for i = 1:T-1
        feedback_control = zeros(input_dim,1);
        for j = 1:i
            % Feedback gain exists for all previous timesteps (causal system)
            feedback_control = feedback_control + K((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim)*(x(:,j) - unstacked_z(:,j));
        end
        % noise = randn(4, 1);     % random vector from N(0,1)
        % noise = noise / norm(noise,1);     % normalize to have norm 1
        noise = disturbance_level * (rand(4,1)*2 - 1);
        x(:,i+1) = A_t*x(:,i) + B_t*(feedback_control + unstacked_v(:,i)) + noise;
        feedback_u(:,i) = feedback_control;
    end
    plot(x(1,1:end),x(2,1:end),"b","DisplayName","Disturbed w/ Feedback Control");
    % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"b","DisplayName","Error signal w/ Feedback Control")
end
xline(b_poly(1),":",'LineWidth',2);
%% Compute phi_x and phi_u
% Then we need to retrieve the phi_x and phi_u matrices from learned K
% C * [phi_x;phi_u] = [I;0]
C = [eye(T*state_dim) - Z*A, -Z*B;
            K              , -eye(input_dim*T)];
Rc = rank(C);
learned_phi = pinv(C)*[eye(state_dim*T);zeros(input_dim*T,state_dim*T)];

% Compare learned phi to ground truth

%% Write the KKT conditions for robust constraints

constraints = [];

