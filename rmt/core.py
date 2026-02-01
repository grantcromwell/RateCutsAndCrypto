import numpy as np
from scipy.linalg import eigh
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

class RandomMatrixTheory:
    """Random Matrix Theory implementation for financial correlation analysis"""

    def __init__(self, N: int, T: int):
        """
        Initialize RMT class

        Args:
            N: Number of assets (dimension of correlation matrix)
            T: Number of time points (number of return observations)
        """
        self.N = N
        self.T = T
        self.Q = T / N  # Matrix aspect ratio

    def marchenko_pastur_distribution(self, x: float) -> float:
        """
        Marchenko-Pastur probability density function

        Args:
            x: Value where to evaluate PDF

        Returns:
            PDF value at x
        """
        gamma_plus = (1 + np.sqrt(1/self.Q))**2
        gamma_minus = (1 - np.sqrt(1/self.Q))**2

        if gamma_minus <= x <= gamma_plus:
            term1 = np.sqrt((gamma_plus - x) * (x - gamma_minus))
            term2 = self.Q / (2 * np.pi * x)
            return term1 * term2
        else:
            return 0.0

    def marchenko_pastur_bounds(self) -> Tuple[float, float]:
        """
        Calculate theoretical bounds of Marchenko-Pastur distribution

        Returns:
            (lambda_min, lambda_max): Lower and upper bounds
        """
        lambda_max = (1 + np.sqrt(self.Q))**2
        lambda_min = (1 - np.sqrt(self.Q))**2 if self.Q <= 1 else 0

        return lambda_min, lambda_max

    def empirical_density(self, eigenvalues: np.ndarray, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute empirical eigenvalue distribution

        Args:
            eigenvalues: Array of eigenvalues
            n_bins: Number of bins for histogram

        Returns:
            (bins, pdf): Bin centers and PDF values
        """
        hist, bin_edges = np.histogram(eigenvalues, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist

    def eigenvalue_spectrum_test(self, eigenvalues: np.ndarray) -> Dict[str, Any]:
        """
        Perform eigenvalue spectrum test against Marchenko-Pastur distribution

        Args:
            eigenvalues: Array of eigenvalues

        Returns:
            Dictionary with test results
        """
        lambda_min_mp, lambda_max_mp = self.marchenko_pastur_bounds()

        # Count eigenvalues outside MP bounds
        signal_eigenvalues = eigenvalues[eigenvalues > lambda_max_mp]
        noise_eigenvalues = eigenvalues[eigenvalues <= lambda_max_mp]

        # Calculate theoretical vs empirical moments
        signal_ratio = len(signal_eigenvalues) / len(eigenvalues)
        noise_ratio = len(noise_eigenvalues) / len(eigenvalues)

        # Calculate eigenvalue statistics
        theoretical_max = lambda_max_mp
        empirical_max = np.max(eigenvalues)
        empirical_min = np.min(eigenvalues)

        return {
            'lambda_min_theoretical': lambda_min_mp,
            'lambda_max_theoretical': lambda_max_mp,
            'lambda_min_empirical': empirical_min,
            'lambda_max_empirical': empirical_max,
            'signal_eigenvalues': len(signal_eigenvalues),
            'noise_eigenvalues': len(noise_eigenvalues),
            'signal_ratio': signal_ratio,
            'noise_ratio': noise_ratio,
            'largest_eigenvalue_ratio': empirical_max / theoretical_max
        }

    def filter_correlation_matrix(self, correlation_matrix: np.ndarray,
                                eigenvalues: np.ndarray,
                                eigenvectors: np.ndarray,
                                threshold_ratio: float = 1.0) -> np.ndarray:
        """
        Filter correlation matrix using eigenvalue thresholding

        Args:
            correlation_matrix: Original correlation matrix
            eigenvalues: Eigenvalues of the matrix
            eigenvectors: Eigenvectors of the matrix
            threshold_ratio: Ratio of lambda_max to use as threshold

        Returns:
            Filtered correlation matrix
        """
        lambda_min_mp, lambda_max_mp = self.marchenko_pastur_bounds()
        threshold = lambda_max_mp * threshold_ratio

        # Identify signal eigenvalues
        signal_mask = eigenvalues > threshold
        noise_mask = ~signal_mask

        # Reconstruct correlation matrix
        # C_filtered = sum_{signal eigenvalues} λ_i * v_i * v_i^T
        correlation_filtered = np.zeros_like(correlation_matrix)

        for i in range(len(eigenvalues)):
            if signal_mask[i]:
                correlation_filtered += eigenvalues[i] * np.outer(eigenvectors[:, i],
                                                             eigenvectors[:, i])

        # Ensure matrix is positive definite and has unit diagonal
        correlation_filtered = self.ensure_valid_correlation(correlation_filtered)

        return correlation_filtered

    def ensure_valid_correlation(self, matrix: np.ndarray) -> np.ndarray:
        """
        Ensure matrix is a valid correlation matrix (positive definite, unit diagonal)

        Args:
            matrix: Input matrix

        Returns:
            Valid correlation matrix
        """
        # Make it symmetric
        matrix = (matrix + matrix.T) / 2

        # Force unit diagonal
        np.fill_diagonal(matrix, 1)

        # Ensure positive definiteness by adding small diagonal if needed
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            # Add small value to diagonal
            min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
            if min_eig < 0:
                matrix += np.eye(matrix.shape[0]) * (-min_eig + 1e-6)

        return matrix

    def compute_participation_ratio(self, eigenvectors: np.ndarray,
                                  eigenvalues: np.ndarray,
                                  idx: int) -> float:
        """
        Calculate participation ratio for eigenvector idx
        Measures how many assets contribute to the eigenvector

        Args:
            eigenvectors: Matrix of eigenvectors
            eigenvalues: Array of eigenvalues
            idx: Index of eigenvector

        Returns:
            Participation ratio (1/N to 1)
        """
        v = eigenvectors[:, idx]
        # Participation ratio: PR = 1 / sum_i |v_i|^4
        pr = 1.0 / np.sum(v**4)
        return pr

    def get_eigenvalue_insights(self, eigenvalues: np.ndarray,
                               eigenvectors: np.ndarray) -> Dict[str, Any]:
        """
        Get insights about eigenvalue structure

        Args:
            eigenvalues: Array of eigenvalues
            eigenvectors: Matrix of eigenvectors

        Returns:
            Dictionary with various insights
        """
        n_eigenvalues = len(eigenvalues)
        results = {}

        # Participation ratios
        participation_ratios = []
        for i in range(n_eigenvalues):
            pr = self.compute_participation_ratio(eigenvectors, eigenvalues, i)
            participation_ratios.append(pr)

        results['participation_ratios'] = participation_ratios
        results['max_participation'] = max(participation_ratios)
        results['min_participation'] = min(participation_ratios)
        results['avg_participation'] = np.mean(participation_ratios)

        # Eigenvalue ratios (hierarchy)
        if n_eigenvalues > 1:
            ratios = eigenvalues[1:] / eigenvalues[:-1]
            results['eigenvalue_ratios'] = ratios
            results['max_ratio'] = np.max(ratios)
            results['min_ratio'] = np.min(ratios)

        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        results['sorted_eigenvalues'] = sorted_eigenvalues
        results['sorted_eigenvectors'] = sorted_eigenvectors

        return results

    def plot_eigenvalue_spectrum(self, eigenvalues: np.ndarray, save_path: str = None):
        """
        Plot eigenvalue spectrum with Marchenko-Pastur distribution

        Args:
            eigenvalues: Array of eigenvalues
            save_path: Path to save the plot
        """
        lambda_min_mp, lambda_max_mp = self.marchenko_pastur_bounds()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Eigenvalue histogram
        x = np.linspace(0, np.max(eigenvalues) * 1.1, 1000)
        pdf_mp = [self.marchenko_pastur_distribution(xi) for xi in x]

        ax1.hist(eigenvalues, bins=20, density=True, alpha=0.7, color='blue',
                label='Empirical distribution')
        ax1.plot(x, pdf_mp, 'r-', linewidth=2, label='Marchenko-Pastur')
        ax1.axvline(lambda_min_mp, color='red', linestyle='--',
                   label=f'λ_min = {lambda_min_mp:.3f}')
        ax1.axvline(lambda_max_mp, color='red', linestyle='--',
                   label=f'λ_max = {lambda_max_mp:.3f}')
        ax1.set_xlabel('Eigenvalue')
        ax1.set_ylabel('Density')
        ax1.set_title('Eigenvalue Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Eigenvalue line plot
        sorted_eig = np.sort(eigenvalues)[::-1]
        ax2.plot(sorted_eig, 'bo-', markersize=8, label='Empirical eigenvalues')
        ax2.axhline(lambda_max_mp, color='red', linestyle='--',
                   label=f'λ_max = {lambda_max_mp:.3f}')
        ax2.set_xlabel('Eigenvalue index (sorted)')
        ax2.set_ylabel('Eigenvalue')
        ax2.set_title('Eigenvalue Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def deflation_analysis(self, correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze eigenvalue deflation pattern

        Args:
            correlation_matrix: Correlation matrix to analyze

        Returns:
            Dictionary with deflation analysis results
        """
        eigenvalues, eigenvectors = eigh(correlation_matrix)

        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        results = {}

        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        results['cumulative_variance'] = cumulative_variance
        results['n_components_95'] = np.argmax(cumulative_variance >= 0.95) + 1

        # Dominant eigenvalue analysis
        results['dominant_eigenvalue'] = eigenvalues[0]
        results['dominant_ratio'] = eigenvalues[0] / np.sum(eigenvalues)

        # Dominant eigenvector analysis
        dominant_vector = eigenvectors[:, 0]
        results['dominant_vector'] = dominant_vector
        results['max_component'] = np.max(np.abs(dominant_vector))
        results['min_component'] = np.min(np.abs(dominant_vector))

        # Check if dominant eigenvector suggests market factor
        is_market_like = results['dominant_ratio'] > 0.5
        results['is_market_like'] = is_market_like

        return results