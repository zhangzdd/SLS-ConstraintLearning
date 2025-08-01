import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are
import math

class CartPoleLQR:
    def __init__(self):
        # Physical parameters of the cartpole system
        # These represent a typical cartpole setup
        self.mass_cart = 1.0      # Mass of cart (kg)
        self.mass_pole = 0.1      # Mass of pole (kg)  
        self.length = 0.5         # Half-length of pole (m)
        self.gravity = 9.81       # Acceleration due to gravity (m/s^2)
        
        # Total mass for calculations
        self.total_mass = self.mass_cart + self.mass_pole
        
        # LQR weight matrices - these determine what we care about most
        # Q matrix: penalizes deviations from desired state (upright, centered)
        # Larger values mean we care more about that state variable
        self.Q = np.diag([10.0,    # Cart position (x) - keep cart near center
                          1.0,     # Cart velocity (x_dot) - don't move too fast
                          100.0,   # Pole angle (theta) - keep pole upright (most important!)
                          1.0])    # Pole angular velocity (theta_dot) - control rotation speed
        
        # R matrix: penalizes control effort (force applied to cart)
        # Higher values mean we prefer gentler control actions
        self.R = np.array([[1.0]])
        
        # Calculate the linearized system matrices around equilibrium (pole upright)
        self.A, self.B = self._get_linear_system_matrices()
        
        # Solve the LQR problem to find optimal feedback gains
        self.K = self._solve_lqr()
        
    def _get_linear_system_matrices(self):
        """
        Linearize the cartpole dynamics around the upright equilibrium.
        
        The full cartpole dynamics are nonlinear, but we can approximate them
        with linear equations for small angles. This makes LQR applicable.
        
        State vector: [x, x_dot, theta, theta_dot]
        where x is cart position, theta is pole angle (0 = upright)
        """
        # These come from linearizing the cartpole equations of motion
        # The derivation involves Lagrangian mechanics, but the key insight is:
        # - Small angle approximation: sin(theta) ≈ theta, cos(theta) ≈ 1
        # - This turns the nonlinear system into a linear one we can control
        
        g = self.gravity
        l = self.length
        mc = self.mass_cart
        mp = self.mass_pole
        mt = self.total_mass
        
        # Denominator term that appears in the linearized dynamics
        denom = mt * l - mp * l  # This simplifies to mc * l for our system
        
        # State matrix A: describes how state evolves over time without control
        A = np.array([
            [0,  1,  0,                    0],                    # x_dot = velocity
            [0,  0,  -mp*g/mc,            0],                    # Cart acceleration
            [0,  0,  0,                    1],                    # theta_dot = angular velocity  
            [0,  0,  mt*g/(mc*l),         0]                     # Pole angular acceleration
        ])
        
        # Input matrix B: describes how control input affects state
        B = np.array([
            [0],           # Control doesn't directly affect position
            [1/mc],        # Force directly accelerates cart (F = ma)
            [0],           # Control doesn't directly affect angle
            [1/(mc*l)]     # Force creates torque on pole through cart motion
        ])
        
        return A, B
    
    def _solve_lqr(self):
        """
        Solve the Linear Quadratic Regulator problem.
        
        LQR finds the optimal feedback gains K that minimize the cost:
        J = integral of (x'Qx + u'Ru) dt
        
        The solution is u = -Kx, where K is computed from the Riccati equation.
        """
        # Solve the continuous-time algebraic Riccati equation
        # This gives us the optimal cost-to-go matrix P
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        
        # The optimal feedback gain matrix K
        # This tells us how much force to apply based on each state variable
        K = np.linalg.inv(self.R) @ self.B.T @ P
        
        print("LQR Gains (K matrix):")
        print(f"  Position gain: {K[0,0]:.3f}")
        print(f"  Velocity gain: {K[0,1]:.3f}") 
        print(f"  Angle gain: {K[0,2]:.3f}")
        print(f"  Angular velocity gain: {K[0,3]:.3f}")
        print()
        
        return K
    
    def get_control_input(self, state):
        """
        Calculate the control input (force) given current state.
        
        This is the heart of LQR: u = -K * x
        The negative sign means we apply force opposite to deviations.
        """
        # State: [x, x_dot, theta, theta_dot]
        # Control input: force applied to cart
        force = -self.K @ state
        return force[0]  # Return scalar force value
    
    def simulate(self, initial_state, time_duration=10.0, dt=0.01):
        """
        Simulate the controlled cartpole system.
        
        Uses simple Euler integration to show how the LQR controller
        stabilizes the system from an initial disturbance.
        """
        # Time vector
        t = np.arange(0, time_duration, dt)
        n_steps = len(t)
        
        # Storage for results  
        states = np.zeros((n_steps, 4))  # [x, x_dot, theta, theta_dot]
        controls = np.zeros(n_steps)     # Applied forces
        
        # Set initial condition
        states[0] = initial_state
        
        print(f"Starting simulation with initial state:")
        print(f"  Cart position: {initial_state[0]:.3f} m")
        print(f"  Cart velocity: {initial_state[1]:.3f} m/s")
        print(f"  Pole angle: {np.degrees(initial_state[2]):.1f} degrees")
        print(f"  Pole angular velocity: {np.degrees(initial_state[3]):.1f} deg/s")
        print()
        
        # Simulation loop
        for i in range(n_steps - 1):
            # Get current state
            x = states[i]
            
            # Calculate optimal control input
            u = self.get_control_input(x)
            controls[i] = u
            
            # Update state using linearized dynamics: x_dot = Ax + Bu
            x_dot = self.A @ x + self.B.flatten() * u
            
            # Simple Euler integration for next state
            states[i+1] = x + x_dot * dt
        
        # Final control input
        controls[-1] = self.get_control_input(states[-1])
        
        return t, states, controls
    
    def animate_cartpole(self, t, states, controls, save_gif=False):
        """
        Create an animated visualization of the cartpole system.
        
        This brings the LQR controller to life by showing how the cart moves
        and the pole sways in response to the control inputs. The animation
        helps build intuition about how the controller responds to disturbances.
        """
        
        # Set up the figure with two subplots: animation and real-time plots
        fig = plt.figure(figsize=(15, 10))
        
        # Main animation subplot (larger)
        ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        
        # Real-time data plots (smaller)
        ax_angle = plt.subplot2grid((3, 2), (2, 0))
        ax_force = plt.subplot2grid((3, 2), (2, 1))
        
        # Animation setup - create a track for the cart to move on
        track_length = 4.0  # Total track length in meters
        ax_main.set_xlim(-track_length/2, track_length/2)
        ax_main.set_ylim(-0.5, 1.2)
        ax_main.set_aspect('equal')
        ax_main.set_title('Cartpole LQR Control Animation', fontsize=16, pad=20)
        ax_main.set_xlabel('Position (m)')
        ax_main.grid(True, alpha=0.3)
        
        # Draw the track (rail that cart slides on)
        track_y = -0.1
        ax_main.plot([-track_length/2, track_length/2], [track_y, track_y], 'k-', linewidth=8, alpha=0.3)
        ax_main.plot([-track_length/2, track_length/2], [track_y, track_y], 'gray', linewidth=6)
        
        # Initialize the cart as a rectangle
        cart_width = 0.3
        cart_height = 0.2
        cart = plt.Rectangle((-cart_width/2, track_y), cart_width, cart_height, 
                           facecolor='blue', edgecolor='black', linewidth=2)
        ax_main.add_patch(cart)
        
        # Initialize the pole as a line
        pole_line, = ax_main.plot([], [], 'r-', linewidth=6, solid_capstyle='round')
        
        # Add a pivot point (joint between cart and pole)
        pivot_point, = ax_main.plot([], [], 'ko', markersize=8)
        
        # Add a mass at the end of the pole
        pole_mass, = ax_main.plot([], [], 'ro', markersize=12)
        
        # Force arrow to show control input
        force_arrow = ax_main.annotate('', xy=(0, 0), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle='->', lw=3, color='green'))
        force_text = ax_main.text(0, 0.8, '', fontsize=12, ha='center', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        # Real-time angle plot setup
        ax_angle.set_xlim(0, t[-1])
        ax_angle.set_ylim(-30, 30)
        ax_angle.set_xlabel('Time (s)')
        ax_angle.set_ylabel('Angle (°)')
        ax_angle.set_title('Pole Angle')
        ax_angle.grid(True, alpha=0.3)
        ax_angle.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        angle_line, = ax_angle.plot([], [], 'r-', linewidth=2)
        
        # Real-time force plot setup  
        max_force = np.max(np.abs(controls)) * 1.1
        ax_force.set_xlim(0, t[-1])
        ax_force.set_ylim(-max_force, max_force)
        ax_force.set_xlabel('Time (s)')
        ax_force.set_ylabel('Force (N)')
        ax_force.set_title('Control Force')
        ax_force.grid(True, alpha=0.3)
        ax_force.axhline(y=0, color='g', linestyle='--', alpha=0.5)
        force_line, = ax_force.plot([], [], 'g-', linewidth=2)
        
        plt.tight_layout()
        
        def animate(frame):
            """
            Animation function called for each frame.
            
            This function updates all the visual elements based on the current
            state of the system. It's like taking a snapshot of the cartpole
            at each moment in time.
            """
            
            # Get current state values
            x_cart = states[frame, 0]      # Cart position
            theta = states[frame, 2]       # Pole angle
            force = controls[frame]        # Applied force
            current_time = t[frame]
            
            # Calculate pole end position using trigonometry
            # The pole rotates around the cart's center
            pole_end_x = x_cart + self.length * np.sin(theta)
            pole_end_y = track_y + cart_height + self.length * np.cos(theta)
            
            # Update cart position
            cart.set_x(x_cart - cart_width/2)
            
            # Update pole (line from cart center to pole end)
            pole_start_x = x_cart
            pole_start_y = track_y + cart_height
            pole_line.set_data([pole_start_x, pole_end_x], [pole_start_y, pole_end_y])
            
            # Update pivot point (where pole connects to cart)
            pivot_point.set_data([pole_start_x], [pole_start_y])
            
            # Update pole mass at the end
            pole_mass.set_data([pole_end_x], [pole_end_y])
            
            # Update force arrow - shows direction and magnitude of applied force
            arrow_scale = 0.02  # Scale factor to make arrow visible
            if abs(force) > 0.1:  # Only show arrow for significant forces
                arrow_start = (x_cart, track_y + cart_height/2)
                arrow_end = (x_cart + force * arrow_scale, track_y + cart_height/2)
                force_arrow.set_position(arrow_end)
                force_arrow.xy = arrow_start
                force_text.set_text(f'Force: {force:.1f} N')
                force_text.set_position((x_cart, 0.9))
            else:
                # Hide arrow when force is small
                force_arrow.set_position((x_cart, track_y + cart_height/2))
                force_arrow.xy = (x_cart, track_y + cart_height/2)
                force_text.set_text('')
            
            # Update real-time plots
            current_angles = np.degrees(states[:frame+1, 2])
            current_forces = controls[:frame+1]
            current_times = t[:frame+1]
            
            angle_line.set_data(current_times, current_angles)
            force_line.set_data(current_times, current_forces)
            
            # Add time display
            time_text = f'Time: {current_time:.2f}s'
            ax_main.set_title(f'Cartpole LQR Control Animation - {time_text}', fontsize=16, pad=20)
            
            return [cart, pole_line, pivot_point, pole_mass, force_arrow, angle_line, force_line]
        
        # Create the animation
        # The interval controls how fast the animation plays
        # frame_skip lets us speed up the animation by skipping frames
        frame_skip = max(1, len(t) // 500)  # Skip frames for reasonable animation speed
        frames_to_animate = range(0, len(t), frame_skip)
        
        print(f"Creating animation with {len(frames_to_animate)} frames...")
        print("Close the plot window when you're done watching the animation.")
        
        anim = FuncAnimation(fig, animate, frames=frames_to_animate, 
                           interval=50, blit=False, repeat=True)
        
        if save_gif:
            print("Saving animation as 'cartpole_lqr.gif'...")
            anim.save('cartpole_lqr.gif', writer='pillow', fps=20)
            print("Animation saved!")
        
        plt.show()
        return anim
    
    def plot_results(self, t, states, controls):
        """Create static plots to analyze the control performance."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Cartpole LQR Control Results', fontsize=14)
        
        # Cart position
        ax1.plot(t, states[:, 0], 'b-', linewidth=2)
        ax1.set_ylabel('Cart Position (m)')
        ax1.set_title('Cart Position vs Time')
        ax1.grid(True, alpha=0.3)
        
        # Pole angle (convert to degrees for readability)
        ax2.plot(t, np.degrees(states[:, 2]), 'r-', linewidth=2)
        ax2.set_ylabel('Pole Angle (degrees)')
        ax2.set_title('Pole Angle vs Time')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Control input (force)
        ax3.plot(t, controls, 'g-', linewidth=2)
        ax3.set_ylabel('Control Force (N)')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Applied Force vs Time')
        ax3.grid(True, alpha=0.3)
        
        # Phase plot: angle vs angular velocity
        ax4.plot(np.degrees(states[:, 2]), np.degrees(states[:, 3]), 'purple', linewidth=2)
        ax4.plot(np.degrees(states[0, 2]), np.degrees(states[0, 3]), 'go', markersize=8, label='Start')
        ax4.plot(np.degrees(states[-1, 2]), np.degrees(states[-1, 3]), 'ro', markersize=8, label='End')
        ax4.set_xlabel('Pole Angle (degrees)')
        ax4.set_ylabel('Angular Velocity (deg/s)')
        ax4.set_title('Phase Plot (Angle vs Angular Velocity)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

# Demonstration: Run the LQR controller with animation
if __name__ == "__main__":
    # Create the LQR controller
    controller = CartPoleLQR()
    
    # Define initial state: [x, x_dot, theta, theta_dot]
    # Start with pole tilted 20 degrees and cart offset from center
    initial_state = np.array([0.1,        # Cart 10cm from center
                             0.0,        # Cart initially at rest
                             np.radians(20),  # Pole tilted 20 degrees
                             0.0])       # Pole initially not rotating
    
    # Run simulation
    print("Running cartpole LQR simulation...")
    time, states, controls = controller.simulate(initial_state, time_duration=8.0)
    
    # Show final state
    final_state = states[-1]
    print(f"Final state after simulation:")
    print(f"  Cart position: {final_state[0]:.4f} m")
    print(f"  Cart velocity: {final_state[1]:.4f} m/s") 
    print(f"  Pole angle: {np.degrees(final_state[2]):.2f} degrees")
    print(f"  Pole angular velocity: {np.degrees(final_state[3]):.2f} deg/s")
    print()
    
    # Check if successfully stabilized (small deviations from upright)
    if abs(final_state[2]) < np.radians(1) and abs(final_state[0]) < 0.1:
        print("✓ Successfully stabilized the cartpole!")
    else:
        print("✗ System not fully stabilized - try adjusting Q and R matrices")
    
    print("\n" + "="*60)
    print("ANIMATION: Watch the LQR controller in action!")
    print("="*60)
    print("The animation shows:")
    print("• Blue cart sliding on the gray track")
    print("• Red pole swaying and being controlled")
    print("• Green arrows showing applied forces")
    print("• Real-time plots of angle and force")
    print("\nObserve how the controller:")
    print("• Responds immediately to the initial disturbance")
    print("• Applies forces that seem to 'push' the cart in the right direction")
    print("• Gradually reduces control effort as the system stabilizes")
    print("• Achieves smooth, optimal control without oscillations")
    print()
    
    # Create the animation - this is where the magic happens!
    anim = controller.animate_cartpole(time, states, controls, save_gif=False)
    
    print("\n" + "="*60)
    print("STATIC ANALYSIS PLOTS")
    print("="*60)
    print("After watching the animation, examine these detailed plots")
    print("to analyze the controller's performance quantitatively.")
    print()
    
    # Show static analysis plots
    controller.plot_results(time, states, controls)
    
    print("\nExperiment suggestions:")
    print("1. Try different initial conditions (larger angles, different positions)")
    print("2. Modify the Q matrix weights to see how it affects control behavior")
    print("3. Change the R value to make the controller more/less aggressive")
    print("4. Add disturbances during simulation to test robustness")