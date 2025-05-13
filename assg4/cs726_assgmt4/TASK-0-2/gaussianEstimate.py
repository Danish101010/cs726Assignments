import numpy as np
import matplotlib.pyplot as plt

def branin_hoo(x):
    x1, x2 = x
    a = 1
    term1 = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
    return a * term1 + term2 + 10

def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    sqdist = np.sum(x1**2, 1).reshape(-1,1) + np.sum(x2**2, 1) - 2*np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    dists = np.sqrt(((x1[:, None, :] - x2[None, :, :])**2).sum(-1))
    sqrt3 = np.sqrt(3)
    return sigma_f**2 * (1 + sqrt3 * dists / length_scale) * np.exp(-sqrt3 * dists / length_scale)

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    sqdist = np.sum(x1**2, 1).reshape(-1,1) + np.sum(x2**2, 1) - 2*np.dot(x1, x2.T)
    return sigma_f**2 * (1 + sqdist / (2 * alpha * length_scale**2)) ** -alpha

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    return -0.5 * y_train.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(x_train) * np.log(2 * np.pi)

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    best_ll = -np.inf
    best_params = (1.0, 1.0)  # length_scale, sigma_f

    for ls in [0.1, 0.5, 1.0, 2.0]:
        for sf in [0.1, 1.0, 5.0]:
            ll = log_marginal_likelihood(x_train, y_train, kernel_func, ls, sf, noise)
            if ll > best_ll:
                best_ll = ll
                best_params = (ls, sf)
    return (*best_params, noise)

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    K_s = kernel_func(x_train, x_test, length_scale, sigma_f)
    K_ss = kernel_func(x_test, x_test, length_scale, sigma_f) + 1e-8 * np.eye(len(x_test))
    
    K_inv = np.linalg.inv(K)
    mu = K_s.T @ K_inv @ y_train
    cov = K_ss - K_s.T @ K_inv @ K_s
    return mu, np.sqrt(np.maximum(np.diag(cov), 1e-10))

def expected_improvement(mu, sigma, y_best, xi=0.01):
    z = (y_best - mu - xi) / sigma
    Phi = 1 / (1 + np.exp(-1.702 * z))
    EI = (y_best - mu - xi) * Phi
    EI[sigma == 0.0] = 0.0
    return EI

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    z = (y_best - mu - xi) / sigma
    Phi = 1 / (1 + np.exp(-1.702 * z))
    PI = Phi
    PI[sigma == 0.0] = 0.0
    return PI


def plot_graph(x1_grid, x2_grid, z_values, x_train, title, filename):
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(x1_grid, x2_grid, z_values, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.scatter(x_train[:, 0], x_train[:, 1], c='red', label='Train Points')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    np.random.seed(0)
    n_samples_list = [10, 20, 50, 100]
    kernels = {
        'rbf': (rbf_kernel, 'RBF'),
        'matern': (matern_kernel, 'Matern (nu=1.5)'),
        'rational_quadratic': (rational_quadratic_kernel, 'Rational Quadratic')
    }
    acquisition_strategies = {
        'EI': expected_improvement,
        'PI': probability_of_improvement
    }
    
    x1_test = np.linspace(-5, 10, 100)
    x2_test = np.linspace(0, 15, 100)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    true_values = np.array([branin_hoo([x1, x2]) for x1, x2 in x_test]).reshape(x1_grid.shape)
    
    for kernel_name, (kernel_func, kernel_label) in kernels.items():
        for n_samples in n_samples_list:
            x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
            y_train = np.array([branin_hoo(x) for x in x_train])
            
            print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
            length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train, kernel_func)
            
            for acq_name, acq_func in acquisition_strategies.items():
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()
                
                y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, 
                                                        kernel_func, length_scale, sigma_f, noise)
                y_mean_grid = y_mean.reshape(x1_grid.shape)
                y_std_grid = y_std.reshape(x1_grid.shape)
                
                if acq_func is not None:
                    # Hint: Find y_best, apply acq_func, select new point, update training set, recompute GP
                    pass
                
                acq_label = '' if acq_name == 'None' else f', Acq={acq_name}'
                plot_graph(x1_grid, x2_grid, true_values, x_train_current,
                          f'True Branin-Hoo Function (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'true_function_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_mean_grid, x_train_current,
                          f'GP Predicted Mean (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_mean_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_std_grid, x_train_current,
                          f'GP Predicted Std Dev (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_std_{kernel_name}_n{n_samples}_{acq_name}.png')

if __name__ == "__main__":
    main()