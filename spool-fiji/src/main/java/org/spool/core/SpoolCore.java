package org.spool.core;

/**
 * Dependency-free core of the SPOOL reconstruction (damped multiplicative
 * Poisson update with an explicit PSF forward operator).
 *
 * Numerically mirrors {@code run_multiemitter_benchmark.py} in the SPOOL
 * repository: zero-padded linear convolution with a centered "same" crop,
 * EPS-regularized ratios, and the damped update A *= (U/C)^eta.
 *
 * Array layout (all row-major / C-order, matching the Python fixture):
 *   Y : double[H*W*T]  photon-count cube, index (h*W + w)*T + t
 *   D : double[K*T]    contrast dictionary (unit-area rows), index k*T + t
 *   A : double[K*H*W]  component amplitude maps, index (k*H + h)*W + w
 */
public final class SpoolCore {

    private SpoolCore() {}

    /** Linear convolution with a centered same-size crop (odd kernels). */
    public static double[] conv2dSame(double[] in, int H, int W,
                                      double[] psf, int ph, int pw) {
        final int y0 = (ph - 1) / 2, x0 = (pw - 1) / 2;
        double[] out = new double[H * W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                double s = 0.0;
                for (int u = 0; u < ph; u++) {
                    int ii = i - u + y0;
                    if (ii < 0 || ii >= H) continue;
                    int rowIn = ii * W, rowK = u * pw;
                    for (int v = 0; v < pw; v++) {
                        int jj = j - v + x0;
                        if (jj < 0 || jj >= W) continue;
                        s += psf[rowK + v] * in[rowIn + jj];
                    }
                }
                out[i * W + j] = s;
            }
        }
        return out;
    }

    /** Adjoint of {@link #conv2dSame}: correlation with the same crop. */
    public static double[] corr2dSame(double[] in, int H, int W,
                                      double[] psf, int ph, int pw) {
        final int y0 = (ph - 1) / 2, x0 = (pw - 1) / 2;
        double[] out = new double[H * W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                double s = 0.0;
                for (int u = 0; u < ph; u++) {
                    int ii = i + u - y0;
                    if (ii < 0 || ii >= H) continue;
                    int rowIn = ii * W, rowK = u * pw;
                    for (int v = 0; v < pw; v++) {
                        int jj = j + v - x0;
                        if (jj < 0 || jj >= W) continue;
                        s += psf[rowK + v] * in[rowIn + jj];
                    }
                }
                out[i * W + j] = s;
            }
        }
        return out;
    }

    private static double initValue(double[] Y, int H, int W, int K) {
        double sum = 0.0;
        for (double y : Y) sum += y;
        return Math.max(sum / ((double) H * W * K), 1e-4);
    }

    /** Pixel-wise Poisson MLE (Dirac-delta PSF limit of the forward model). */
    public static double[] mle(double[] Y, double[] D,
                               int H, int W, int T, int K,
                               double bg, int nIter, double eps) {
        double[] Dsum = new double[K];
        for (int k = 0; k < K; k++) {
            double s = 0.0;
            for (int t = 0; t < T; t++) s += D[k * T + t];
            Dsum[k] = s;
        }
        double[] A = new double[K * H * W];
        java.util.Arrays.fill(A, initValue(Y, H, W, K));
        double[] U = new double[K * H * W];

        for (int it = 0; it < nIter; it++) {
            java.util.Arrays.fill(U, 0.0);
            for (int p = 0; p < H * W; p++) {
                int yBase = p * T;
                for (int t = 0; t < T; t++) {
                    double lam = bg;
                    for (int k = 0; k < K; k++) lam += A[k * H * W + p] * D[k * T + t];
                    double r = Y[yBase + t] / (lam + eps);
                    for (int k = 0; k < K; k++) U[k * H * W + p] += D[k * T + t] * r;
                }
            }
            for (int k = 0; k < K; k++) {
                double norm = Dsum[k] + eps;
                int base = k * H * W;
                for (int p = 0; p < H * W; p++) A[base + p] *= U[base + p] / norm;
            }
        }
        return A;
    }

    /** SPOOL joint spatial-contrast reconstruction. */
    public static double[] joint(double[] Y, double[] D, double[] psf,
                                 int H, int W, int T, int K, int ph, int pw,
                                 double bg, int nIter, double eta, double eps) {
        double[] ones = new double[H * W];
        java.util.Arrays.fill(ones, 1.0);
        double[] convOnes = conv2dSame(ones, H, W, psf, ph, pw);
        double[] Dsum = new double[K];
        for (int k = 0; k < K; k++) {
            double s = 0.0;
            for (int t = 0; t < T; t++) s += D[k * T + t];
            Dsum[k] = s;
        }

        double[] A = new double[K * H * W];
        java.util.Arrays.fill(A, initValue(Y, H, W, K));
        double[] Ac = new double[K * H * W];
        double[] U1 = new double[K * H * W];
        double[] plane = new double[H * W];

        for (int it = 0; it < nIter; it++) {
            for (int k = 0; k < K; k++) {
                System.arraycopy(A, k * H * W, plane, 0, H * W);
                double[] c = conv2dSame(plane, H, W, psf, ph, pw);
                System.arraycopy(c, 0, Ac, k * H * W, H * W);
            }
            java.util.Arrays.fill(U1, 0.0);
            for (int p = 0; p < H * W; p++) {
                int yBase = p * T;
                for (int t = 0; t < T; t++) {
                    double lam = bg;
                    for (int k = 0; k < K; k++) lam += Ac[k * H * W + p] * D[k * T + t];
                    double r = Y[yBase + t] / (lam + eps);
                    for (int k = 0; k < K; k++) U1[k * H * W + p] += D[k * T + t] * r;
                }
            }
            for (int k = 0; k < K; k++) {
                System.arraycopy(U1, k * H * W, plane, 0, H * W);
                double[] u = corr2dSame(plane, H, W, psf, ph, pw);
                int base = k * H * W;
                for (int p = 0; p < H * W; p++) {
                    double c = convOnes[p] * Dsum[k] + eps;
                    double ratio = u[p] / c;
                    if (ratio < 0.0) ratio = 0.0;
                    A[base + p] *= Math.pow(ratio, eta);
                }
            }
        }
        return A;
    }

    /** Amplitude-weighted parameter map. */
    public static double[] parameterMap(double[] A, double[] thetas,
                                        int K, int H, int W, double eps) {
        double[] out = new double[H * W];
        for (int p = 0; p < H * W; p++) {
            double tot = 0.0, num = 0.0;
            for (int k = 0; k < K; k++) {
                double a = A[k * H * W + p];
                tot += a;
                num += thetas[k] * a;
            }
            out[p] = tot > eps ? num / tot : 0.0;
        }
        return out;
    }
}
