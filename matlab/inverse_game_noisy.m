% Load the forward game data
% clf;clear;
% load("forward_game_data.mat");

%%
% Time to implement the inverse game!
% First we need to invert K from the demonstrations



inverse_run = 20;

transmission_noise_level = {0.001,0.005,0.01,0.05,0.1,0.5};
theta_error = cell(length{transmission_noise_level}, inverse_run);

for noise_id = 1:length{transmission_noise_level}
    for run_id = 1:inverse_run
    
        X = zeros(state_dim*T,num_rollout-1);
        U = zeros(input_dim*T,num_rollout-1);
        
        % noise = disturbance_level * (rand(4,1)*2 - 1);
    
        
        
        
        for t = 1:T
            x_diff = {};
            u_diff = {};
            x_diff_unstacked = zeros(state_dim,num_rollout-1);
            u_diff_unstacked = zeros(input_dim,num_rollout-1);
            for rollout_cnt = 1:num_rollout-1
                x_diff{rollout_cnt} = state_trajectory_closedloop{rollout_cnt+1}(:,t) - state_trajectory_closedloop{rollout_cnt}(:,t) + transmission_noise_level{noise_id} * (rand(state_dim,1)*2 - 1);
                u_diff{rollout_cnt} = input_trajectory_closedloop{rollout_cnt+1}(:,t) - input_trajectory_closedloop{rollout_cnt}(:,t) + transmission_noise_level{noise_id} * (rand(input_dim,1)*2 - 1);
                x_diff_unstacked(:,rollout_cnt) = x_diff{rollout_cnt};
                u_diff_unstacked(:,rollout_cnt) = u_diff{rollout_cnt};
            end
            X((t-1)*state_dim+1:t*state_dim,:) = x_diff_unstacked;
            U((t-1)*input_dim+1:t*input_dim,:) = u_diff_unstacked;
        end
        
        % This is supposed to a lower triangular matrix
        K = U * pinv(X);
        % Compare against the ground truth
        is_close = norm(K - phi_u / phi_x, 'fro') < 1e-5
        %% Roll out with learned K
        % figure(3); hold on;
        % for rollout_cnt = 1:num_rollout
        %     % Roll out trajectory with noise to the dynamics, WITHOUT the feedback
        %     x_init = [0;0;0;0]; % same as nominal trajectory initial state
        %     x = zeros(state_dim,T);
        % 
        %     for i = 1:T-1
        %         % noise = randn(4, 1);     % random vector from N(0,1)
        %         noise = disturbance_level * (rand(4,1)*2 - 1);
        %         new_x = A_t*x(:,i) + B_t*unstacked_v(:,i) + noise;
        %         x(:,i+1) = new_x;
        %     end
        %     plot(x(1,:),x(2,:),"r","DisplayName","Disturbed");
        %     % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"r","DisplayName","Error signal without Feedback Control")    
        %     % Roll out trajectory with noise to the dynamics, WITH the feedback
        % 
        %     x_init = [0;0;0;0];
        %     x = zeros(state_dim,T);
        %     feedback_u = zeros(input_dim,T);
        % 
        %     for i = 1:T-1
        %         feedback_control = zeros(input_dim,1);
        %         for j = 1:i
        %             % Feedback gain exists for all previous timesteps (causal system)
        %             feedback_control = feedback_control + K((i-1)*input_dim+1:i*input_dim,(j-1)*state_dim+1:j*state_dim)*(x(:,j) - unstacked_z(:,j));
        %         end
        %         % noise = randn(4, 1);     % random vector from N(0,1)
        %         % noise = noise / norm(noise,1);     % normalize to have norm 1
        %         noise = disturbance_level * (rand(4,1)*2 - 1);
        %         x(:,i+1) = A_t*x(:,i) + B_t*(feedback_control + unstacked_v(:,i)) + noise;
        %         feedback_u(:,i) = feedback_control;
        %     end
        %     plot(x(1,1:end),x(2,1:end),"b","DisplayName","Disturbed w/ Feedback Control");
        %     % plot(x(1,1:end) - unstacked_z(1,1:end),x(2,1:end) - unstacked_z(2,1:end),"b","DisplayName","Error signal w/ Feedback Control")
        % end
        % xline(b_poly(1),":",'LineWidth',2);
        %% Compute phi_x and phi_u
        % Then we need to retrieve the phi_x and phi_u matrices from learned K
        % C * [phi_x;phi_u] = [I;0]
        C = [eye(T*state_dim) - Z*A, -Z*B;
                    K              , -eye(input_dim*T)];
        Rc = rank(C);
        learned_phi = pinv(C)*[eye(state_dim*T);zeros(input_dim*T,state_dim*T)];
        learned_phi_x = learned_phi(1:T*state_dim,:);
        learned_phi_u = learned_phi(T*state_dim+1:end,:);
        % Compare learned phi to ground truth
        norm(learned_phi_x - phi_x, 'fro') 
        norm(learned_phi_u - phi_u, 'fro')
        
        
        %% Solve for the nominal trajectory, problem is formulated into a linear program
        
        z_norm = sdpvar(T*state_dim,1);
        v_norm = sdpvar(T*input_dim,1);
        epsilon = 1e-5;
        residual = 0;
        constraints = [];
        
        for rollout_cnt = 1:num_rollout
            for t = 1:T
                feedback_term = 0;
                for j = 1:t
                    % Feedback gain exists for all previous timesteps (causal system)
                    feedback_term = feedback_term + K((t-1)*input_dim+1:t*input_dim,(j-1)*state_dim+1:j*state_dim)*(state_trajectory_closedloop{rollout_cnt}(:,j) - z_norm((j-1)*state_dim+1:j*state_dim));
                end
                u_t = feedback_term;
                residual = residual + norm(input_trajectory_closedloop{rollout_cnt}(:,t) - u_t, 1);
        
                % constraints = [constraints, -epsilon <= input_trajectory_closedloop{rollout_cnt}(:,t) - u_t, input_trajectory_closedloop{rollout_cnt}(:,t) - u_t <= epsilon];
                if t<=T-1
                    constraints = [constraints, -epsilon <= z_norm(t*state_dim+1:(t+1)*state_dim) - A_t*z_norm((t-1)*state_dim+1:t*state_dim) - B_t*v_norm((t-1)*input_dim+1:t*input_dim) <= epsilon];
                    residual = residual + norm(z_norm(t*state_dim+1:(t+1)*state_dim) - A_t*z_norm((t-1)*state_dim+1:t*state_dim) - B_t*v_norm((t-1)*input_dim+1:t*input_dim),1);
                end
            end
        end
        
        % Sanity check: enforce the z_norm to be groundtruth z
        % constraints = [constraints, z_norm == unstacked_z(:)];
        
        % More prior knowledge
        constraints = [constraints, z_norm(end-3:end) == unstacked_z(:,end)];
        constraints = [constraints, z_norm(1:4) == unstacked_z(:,1)];
        
        % constraints = [constraints, z_norm((T-1)*state_dim+3:T*state_dim) == [0;0]];
        
        % z_norm((T-1)*state_dim:T*state_dim)
        ops = sdpsettings('solver','gurobi','verbose', 2);
        % Feasibility problem
        optimization_results = optimize(constraints, residual, ops);
        
        
        %% Extract the closest nominal trajectory
        
        unstacked_z = reshape(value(z_norm),[state_dim, T]);
        unstacked_v = reshape(value(v_norm),[input_dim, T]);
        
        
        %% Write the KKT conditions for robust-constraints-learning
        
        constraints = [];
        % Nominal trajectory z,v
        z = sdpvar(state_dim*T,1);
        v = sdpvar(input_dim*T,1);
        % A_poly = sdpvar(1,4);
        A_poly = [1 0 0 0];
        b_poly = sdpvar(1,1);
        
        lambda_affine = sdpvar(T,1);
        constraint_tight = binvar(T,2);
        M = 1e5;
        grad_affine = cell(T,1);
        grad_dyn = cell(T-1,1);
        lambda_dyn = sdpvar(T-1, state_dim,'full');
        affine_residual = cell(T,1);
        slack = sdpvar(T,1);
        
        constrained_states_upper_limited = 4;
        for dim = 1:DimAffine
            for k = 1:T-1
                phi_kx = phi_x((k-1)*state_dim+1:k*state_dim,:);
                phi_ku = phi_u((k-1)*input_dim+1:k*input_dim,:);
                z_k = z((k-1)*state_dim+1:k*state_dim);
                v_k = v((k-1)*input_dim+1:k*input_dim);
                z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
                dyn_residual = z_k_plus_1 - A_t*z_k - B_t*v_k;
        
                grad_dyn{k} = jacobian(dyn_residual,z);
        
         
                constraints = [constraints, replace(dyn_residual,z,unstacked_z(:)) == 0];
                robust_affine_constraints = 0;
                for j = 1:k
                    robust_affine_constraints = robust_affine_constraints + ...
                        disturbance_level * norm(A_poly(dim,:)*phi_kx(:,(j-1)*state_dim+1:j*state_dim),1);
                end
                % constraints = [constraints, A_poly(dim,:)*z_k + robust_affine_constraints <= b_poly(dim)];
                affine_residual{k} = A_poly(dim,:)*z_k + robust_affine_constraints - b_poly(dim);
               
        
                % Primal feasibility
                constraints = [constraints, replace(affine_residual{k},z,unstacked_z(:)) <= 0];
                
                % Complementary slackness
                constraints = [constraints, replace(affine_residual{k},z,unstacked_z(:)) >= -M*constraint_tight(k,1)];
                constraints = [constraints, 0 <= lambda_affine(k), lambda_affine(k) <= M*constraint_tight(k,2)];
                constraints = [constraints, sum(constraint_tight(k,:))<=1];
                
                % Stationarity contribution
                grad_affine{k} = jacobian(affine_residual{k},[z;v]);
        
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
        
        % In the inverse game we dont know the exact terminal/initial constraints
        % constraints = [constraints, z_init(1:state_dim) == [0;0;0;0]];
        % constraints = [constraints, z_terminal(1:state_dim) == [2.9;20;0;0]];
        % constraints = [constraints, v_terminal(1:input_dim) == [0;0]];
        
        % Stationarity contribution of terminal constraints
        grad_term = jacobian(z_terminal,z);
        grad_init = jacobian(z_init,z);
        lambda_term = sdpvar(1,4);
        lambda_init = sdpvar(1,4);
        
        
        % Terminal robust constraint
        for dim = 1:DimAffine
            robust_affine_constraints = 0;
            for j = 1:T
                robust_affine_constraints = robust_affine_constraints + disturbance_level * ...
                    norm(A_poly(dim,:)*phi_x((T-1)*state_dim+1:T*state_dim,(j-1)*state_dim+1:j*state_dim),1);
            end
            affine_residual{T} = A_poly(dim,:)*z_terminal + robust_affine_constraints - b_poly(dim);
        
            % Primal feasibility
            constraints = [constraints, replace(affine_residual{T},z,unstacked_z(:)) <= 0];
        
            % Complementary slackness
            constraints = [constraints, replace(affine_residual{T},z,unstacked_z(:)) >= -M*constraint_tight(T,1)];
            constraints = [constraints, 0 <= lambda_affine(T), lambda_affine(T) <= M*constraint_tight(T,2)];
            constraints = [constraints, sum(constraint_tight(T,:))<=1];
            grad_affine{T} = jacobian(affine_residual{T},[z;v]);
        
        end
        
        
        % SLS constraints
        % constraints = [constraints, [eye(T*state_dim) - Z*A, -Z*B]*[phi_x;phi_u] == eye(T*state_dim)];
        
        
        % Objective
        objective = 0;
        for k = 1:T-1
            z_k = z((k-1)*state_dim+1:k*state_dim);
            z_k_plus_1 = z(k*state_dim+1:(k+1)*state_dim);
            
            objective = objective + sum((z_k_plus_1(1:2) - z_k(1:2)).^2) - z_k(1);
        end
        
        grad_obj = jacobian(objective, [z;v]);
        
        % dynamics jacobians
        bicycle_dyn_f = @(x) [x(1),x(2),x(3),x(4)]';
        DT = 0.1;
        bicycle_dyn_g = @(x,u) [0 0 1 0;0 0 0 1; 0 0 0 0; 0 0 0 0]*x + [0 0;0 0;1 0;0 1]*u;
        
        
        traj = unstacked_z;
        u_traj = unstacked_v;
        traj_var_sym = sym('traj_var', size(traj));
        u_traj_var_sym = sym('u_traj_var', size(u_traj));
        
        nu_var_dyn_term = []; % the collection of constraints residual
        for i = 1:size(traj, 2)-1
        nu_var_dyn_term = [nu_var_dyn_term, traj_var_sym(:, i+1) - ( traj_var_sym(:, i) + ...
          bicycle_dyn_g(traj_var_sym(:,i),u_traj_var_sym(:, i))*DT)];
        end
        nu_var_dyn_jac = jacobian(nu_var_dyn_term(:), [traj_var_sym(:); u_traj_var_sym(:)]);
        dyn_jacobians = double(subs(nu_var_dyn_jac, [traj_var_sym(:); u_traj_var_sym(:)], [traj(:); u_traj(:)]));
        
        
        % Collect the stationarity terms
        stationarity = 0;
        for t = 1:T
            stationarity = stationarity + grad_affine{t} * lambda_affine(t); 
        end
        nu_var_dyn = sdpvar(1, size(dyn_jacobians, 1)); %nx * T-1
        nu_dyn = nu_var_dyn * dyn_jacobians;
        stationarity = stationarity + nu_dyn;
        nu_var1 = sdpvar(1, 2*size(traj,1) );
        nu_term1 = [nu_var1(1:size(traj, 1) ), zeros(1, length(traj(:)) - 2*size(traj, 1)), ...
        nu_var1(size(traj, 1)+1:end ), zeros(1, length(u_traj(:)))];
        nu_term = nu_term1;
        stationarity = stationarity + grad_obj + nu_term;
        
        % Plug in the solved nominal trajectory
        % numerical_stationarity = replace(stationarity, z, z_norm(:));
        numerical_stationarity = replace(stationarity, z, unstacked_z(:));
        % Solve
        constraints = [constraints, slack >= 0];
        
        % Sanity checks
        % constraints = [constraints, b_poly <= 4.99]; % This should give non-zero stationarity
        % constraints = [constraints, b_poly >= 5.01]; % This should set the problem to be infeasible
        
        ops = sdpsettings('solver','gurobi','verbose', 2);
        optimization_results = optimize(constraints, norm(numerical_stationarity,1), ops);
        theta_error{noise_id,run_id} = value(b_poly) - 5;

        % Save at checkpoints
        if rem(run_id,5) == 0
            save(sprintf("noisy_inversion_%d_%02d.mat",noise_id,run_id),"theta_error","transmission_noise_level");
        end
    end
end


