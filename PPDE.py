import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras.backend as K
from tensorflow.keras.layers import Concatenate
from keras.layers import Dense, Activation,LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.optimizers import Adam
from keras.regularizers import L1L2
import time as ttt  # to avoid conflict with 'time' placeholder
import matplotlib.pyplot as plt
import numpy as np

def generate_t(T, steps, M, dt):
    '''
    time discretization (M * (steps+1 +1)) 
    for computing the time derivative, we need an extra time step.
    '''
    t_temp = np.linspace(1e-6, T- 1e-6, steps +1, dtype = np.float32)
    return np.tile(np.concatenate((t_temp, [T + dt])), (M,1)) # extra after terminal
    
def Create_paths(i, M, x_0, r, q, sigma, T, steps):
    '''
    Generate GBM paths (M × (steps+1)) with seed `i`.
    
    Arguments:
    - i: random seed
    - M: number of sample paths
    - x_0: initial value
    - r: risk-free rate
    - q: dividend yield
    - sigma: volatility
    - T: terminal time
    - steps: number of time steps
    
    Returns:
    - path_selection: selected GBM paths (M × (steps+1))
    '''
    np.random.seed(i)
    x = np.tile(x_0, (M, 1))
    path = np.tile(x_0, (M, 1))
    delta = T / 1000

    dW = np.sqrt(delta) * np.random.normal(size=(M, 1000))
    for k in range(1000):
        x += (r - q) * x * delta + sigma * x * dW[:, k:k+1]
        path = np.concatenate((path, x), axis=1)

    # Select downsampled steps
    selection = np.linspace(0, 1000, steps + 1, dtype=np.int32)
    path_selection = path[:, selection]

    return np.array(path_selection, dtype=np.float32)


############################################## LOSS FUNCTION #####################################
############################################################################################

def loss_function(time, path, M, payoff_fn, r, q, sigma, dt, steps):
    ################################ Model: Feedforward Neural Network ###########################
    NN = Sequential([
        Dense(128, input_shape=(2,)),  # 2 = (space, time)
        Activation('tanh'),
        Dense(128),
        Activation('tanh'),
        Dense(128),
        Activation('tanh'),
        Dense(1)
    ])

    ################################ Initial Step ################################################
    ## Approximate f(X_{t_0})
    input_x = tf.slice(path, [0, 0], [M, 1])
    input_t = tf.slice(time, [0, 0], [M, 1])
    inputt_f = Concatenate(axis=-1)([input_x, input_t])
    f = NN(inputt_f)

    ## Calculate space derivatives
    bump = 0.01 * input_x
    input_x_up = input_x + bump
    inputt_up = Concatenate(axis=-1)([input_x_up, input_t])
    f_up = NN(inputt_up)

    input_x_down = input_x - bump
    inputt_down = Concatenate(axis=-1)([input_x_down, input_t])
    f_down = NN(inputt_down)

    partial_x_f = (f_up - f) / bump
    partial_xx_f = (f_up - 2 * f + f_down) / (bump ** 2)

    ## Calculate time derivative
    input_t_time = tf.slice(time, [0, 1], [M, 1])
    inputt_time = Concatenate(axis=-1)([input_x, input_t_time])
    f_flat = NN(inputt_time)
    partial_t_f = (f_flat - f) / dt

    ## First PDE residual loss
    Loss = tf.reduce_sum(tf.square(
        partial_t_f + (r - q) * input_x * partial_x_f + 0.5 * sigma ** 2 * input_x ** 2 * partial_xx_f - r * f
    ))

    ## Save solution and derivatives
    solution = f
    time_derivative = partial_t_f
    space_derivative = partial_x_f
    space_2nd_derivative = partial_xx_f

    ################################ Forward Propagation Over Time ##############################
    for i in range(1, steps + 1):
        input_x = tf.slice(path, [0, i], [M, 1])
        input_t = tf.slice(time, [0, i], [M, 1])
        inputt_f = Concatenate(axis=-1)([input_x, input_t])
        f = NN(inputt_f)

        bump = 0.01 * input_x
        input_x_up = input_x + bump
        inputt_up = Concatenate(axis=-1)([input_x_up, input_t])
        f_up = NN(inputt_up)

        input_x_down = input_x - bump
        inputt_down = Concatenate(axis=-1)([input_x_down, input_t])
        f_down = NN(inputt_down)

        partial_x_f = (f_up - f) / bump
        partial_xx_f = (f_up - 2 * f + f_down) / (bump ** 2)

        input_t_time = tf.slice(time, [0, i + 1], [M, 1])
        inputt_time = Concatenate(axis=-1)([input_x, input_t_time])
        f_flat = NN(inputt_time)
        partial_t_f = (f_flat - f) / dt

        Loss += tf.reduce_sum(tf.square(
            partial_t_f + (r - q) * input_x * partial_x_f + 0.5 * sigma ** 2 * input_x ** 2 * partial_xx_f - r * f
        ))

        solution = Concatenate(axis=-1)([solution, f])
        time_derivative = Concatenate(axis=-1)([time_derivative, partial_t_f])
        space_derivative = Concatenate(axis=-1)([space_derivative, partial_x_f])
        space_2nd_derivative = Concatenate(axis=-1)([space_2nd_derivative, partial_xx_f])

    #############################################################################
    ## Terminal cost: Injected via payoff_fn
    terminal_payoff = payoff_fn(path)
    Loss += tf.reduce_sum(tf.square(f - terminal_payoff)) * steps
    return Loss / M / steps, solution, time_derivative, space_derivative, space_2nd_derivative
    

############################################## TRAINING #####################################
############################################################################################

def train_ppde_model(
    sess,
    loss_function,
    Create_paths,
    generate_t,
    true_solution,
    terminal_condition,
    time,
    path,
    T,
    M,
    dt,
    steps,
    Epoch,
    clip_norm,
    learning_rate_start
):

    # --- Loss and Derivatives ---
    loss, solution, time_derivative, space_derivative, space_2nd_derivative = loss_function(time, path, M)

    # --- Learning Rate and Optimizer ---
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.maximum(
        tf.train.exponential_decay(learning_rate_start, global_step, 50, 0.98, staircase=True),
        tf.constant(0.00001)
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)


    # --- Initialize ---
    sess.run(tf.global_variables_initializer())

    np.random.seed(8)
    path_test = Create_paths(100000000, M)
    time_feed = generate_t(T, steps, M, dt)
    pred_dict = {path: path_test, time: time_feed}

    train_loss_list = []
    test_loss_list = []
    t_axis = np.linspace(0, T, steps + 1)

    def log_and_plot(epoch, loss_val, loss_test_val, lr_val, elapsed):
        print(f"{epoch}th Epoch:")
        print(f"training loss: {loss_val:.5f}, test loss: {loss_test_val:.5f}, "
              f"learning rate: {lr_val:.5f}, elapsed: {elapsed:.2f}s\n")
        train_loss_list.append(loss_val)
        test_loss_list.append(loss_test_val)

    def plot_results():
        num_epochs = len(train_loss_list)
        x_epochs = np.linspace(0, num_epochs * 10, num_epochs)

        # Loss Plot
        plt.plot(x_epochs, train_loss_list, 'b-', label="train loss")
        plt.plot(x_epochs, test_loss_list, 'c--', label="test loss")
        plt.legend()
        plt.ylim([0, 0.1])
        plt.title("Total Loss")
        plt.show()

        # Example Path
        plt.plot(t_axis, path_test[1], label="GBM Path")
        plt.legend()
        plt.title("One Test Path")
        plt.show()

        # Solution vs True
        solution_pred, time_der, space_der, space2_der = sess.run(
            [solution, time_derivative, space_derivative, space_2nd_derivative],
            pred_dict
        )
        plt.plot(t_axis, true_solution(path_test[1]), label="True")
        plt.plot(t_axis, solution_pred[1], 'c--', label="Predicted")
        plt.plot(T, terminal_condition(path_test[1]), "ro", label="Terminal Value")
        plt.legend()
        plt.title("Corresponding Solution")
        plt.show()

        # Derivatives Plot
        plt.plot(t_axis, space_der[1], label="space derivative")
        plt.plot(t_axis, space2_der[1], label="2nd space derivative")
        plt.plot(t_axis, time_der[1], label="time derivative")
        plt.legend()
        plt.title("Derivatives")
        plt.show()

    # --- Training Loop ---
    start_time = ttt.time()
    for it in range(Epoch):
        seed = it % 100
        path_feed = Create_paths(seed, M)
        feed_dict = {path: path_feed, time: time_feed}
        sess.run(train_op, feed_dict)

        if it % 10 == 0:
            elapsed = ttt.time() - start_time
            loss_val = sess.run(loss, feed_dict)
            loss_test_val = sess.run(loss, pred_dict)
            lr_val = sess.run(learning_rate)
            log_and_plot(it + 1, loss_val, loss_test_val, lr_val, elapsed)
            start_time = ttt.time()

        #if it % 100 == 0:
        #    plot_results()

    return solution, time_derivative, space_derivative, space_2nd_derivative


############################################## Visualization #####################################
############################################################################################

import os

def visualize_model_output(
    sess,
    solution,
    time_derivative,
    space_derivative,
    space_2nd_derivative,
    true_solution_fn,
    terminal_condition_fn,
    Create_paths,
    generate_t,
    T,
    M,
    dt,
    steps,
    path,
    time,
    option_name="Generic Option"
):
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure the directory exists
    save_dir = "/content/numerical_results/BS3"
    os.makedirs(save_dir, exist_ok=True)

    # Generate time grid
    t = np.linspace(0, T, steps + 1)
    time_feed = generate_t(T, steps, M, dt)

    # Generate random test paths
    np.random.seed(8)
    path_test = Create_paths(100000000, M)
    pred_dict = {path: path_test, time: time_feed}

    # Evaluate model outputs
    solution_pred, time_der_pred, space_der_pred, space2_der_pred = sess.run(
        [solution, time_derivative, space_derivative, space_2nd_derivative],
        pred_dict
    )

    # === Plot 16 sample GBM paths ===
    plt.figure(figsize=(8, 4))
    for i in range(16):
        plt.plot(t, path_test[i])
    plt.title(f"Sample GBM Paths ({option_name})")
    plt.xlabel("t")
    plt.ylabel(r"$Y_t$")
    plt.savefig(f"{save_dir}/sample_gbm_paths_{option_name}.png")
    plt.close()

    # === Plot solution and derivatives for a few paths ===
    test_indices = [2]  # Extend this list for more test cases

    for idx in test_indices:
        # Plot sample path
        plt.plot(t, path_test[idx], label=f"Test Path {idx}")
        plt.xlabel("t")
        plt.ylabel(r"$Y_t$")
        plt.legend()
        plt.title(f"{option_name}: Sample Test Path {idx}")
        plt.savefig(f"{save_dir}/path_{idx}_yt_{option_name}.png")
        plt.close()

        # True vs Predicted solution
        plt.plot(t, true_solution_fn(path_test[idx]), label="True Solution", color="blue")
        plt.plot(t, solution_pred[idx], label="Predicted Solution", linestyle='--', color="red")
        plt.plot(T, terminal_condition_fn(path_test[idx]), "ko", label="Terminal Value")
        plt.xlabel("t")
        plt.ylabel(r"$f(Y_t)$")
        plt.title(f"{option_name}: Solution Comparison (Test Path {idx})")
        plt.legend()
        plt.savefig(f"{save_dir}/solution_comparison_{idx}_{option_name}.png")
        plt.close()

        # Optional: save derivatives plot (uncomment if needed)
        # plt.plot(t, space_der_pred[idx], label="space derivative")
        # plt.plot(t, space2_der_pred[idx], label="2nd space derivative")
        # plt.plot(t, time_der_pred[idx], label="time derivative")
        # plt.title(f"{option_name}: Derivatives for Test Path {idx}")
        # plt.xlabel("t")
        # plt.ylabel("Derivative values")
        # plt.legend()
        # plt.savefig(f"{save_dir}/derivatives_{idx}_{option_name}.png")
        # plt.close()


    # Optional: visualize synthetic paths
    """def show_synthetic_path(path_array, label, idx=1):
        pred_dict_custom = {path: path_array, time: time_feed}
        sol, s_der, t_der = sess.run([solution, space_derivative, time_derivative], pred_dict_custom)

        # Plot path
        plt.plot(t, path_array[idx], label=label)
        plt.xlabel("t")
        plt.ylabel(r"$Y_t$")
        plt.legend()
        plt.title(f"{option_name}: {label}")
        plt.show()

        # Plot solution
        plt.plot(t, true_solution_fn(path_array[idx]), label="True", color="green")
        plt.plot(t, sol[idx], "r--", label="Predicted")
        plt.plot(T, terminal_condition_fn(path_array[idx]), "ko", label="Terminal Value")
        plt.title(f"{option_name}: Solution on {label}")
        plt.xlabel("t")
        plt.ylabel(r"$f(Y_t)$")
        plt.legend()
        plt.show()

    # Example synthetic test paths
    decreasing_path = np.tile(np.linspace(1, 1e-6, steps + 1), (M, 1)) ** 2
    show_synthetic_path(decreasing_path, "Decreasing Path")

    np.random.seed(5)
    uniform_path = np.tile(np.random.uniform(1, 3, steps + 1), (M, 1))
    show_synthetic_path(uniform_path, "Uniform Random Path")

    # Compare all 3 paths
    plt.plot(t, path_test[2], label="Test Path 1")
    plt.plot(t, decreasing_path[1], label="Decreasing Path")
    plt.plot(t, uniform_path[1], label="Uniform Random Path")
    plt.xlabel("t")
    plt.ylabel(r"$Y_t$")
    plt.title(f"{option_name}: Comparison of Test Paths")
    plt.legend()
    plt.show()
    """



