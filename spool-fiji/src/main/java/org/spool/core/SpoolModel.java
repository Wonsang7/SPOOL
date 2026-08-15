package org.spool.core;

import org.apache.commons.math3.special.Erf;

/** Forward-model builders for the SPOOL Fiji plugin. */
public final class SpoolModel {

    private SpoolModel() {}

    public static final double FWHM_TO_SIGMA = 2.3548200450309493;

    /** Pixel-integrated unit-sum Gaussian PSF; radius = ceil(4*sigma), odd size. */
    public static double[] gaussianPsf(double sigmaPx, int[] sizeOut) {
        int r = Math.max((int) Math.ceil(4.0 * sigmaPx), 1);
        int n = 2 * r + 1;
        double[] w = new double[n];
        double s2 = Math.sqrt(2.0) * sigmaPx;
        for (int i = 0; i < n; i++) {
            double a = i - r;
            w[i] = 0.5 * (Erf.erf((a + 0.5) / s2) - Erf.erf((a - 0.5) / s2));
        }
        double[] p = new double[n * n];
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) { p[i * n + j] = w[i] * w[j]; sum += p[i * n + j]; }
        for (int i = 0; i < p.length; i++) p[i] /= sum;
        if (sizeOut != null) { sizeOut[0] = n; sizeOut[1] = n; }
        return p;
    }

    /**
     * Estimate the IRF center t0 (in time-bin units) directly from a photon-count cube.
     *
     * The fluorescence maximum itself is lifetime-dependent and is therefore a biased
     * proxy for the IRF center. Instead, this estimator sums all pixels to obtain a
     * high-SNR global decay, applies a five-point binomial smoothing kernel, and finds
     * the strongest positive slope on the leading edge. A quadratic interpolation of
     * the derivative gives a sub-bin estimate.
     *
     * Y layout: ((y * W) + x) * T + t.
     */
    public static double estimateIrfPeakBin(double[] Y, int H, int W, int T) {
        if (T <= 0) throw new IllegalArgumentException("T must be positive");

        final double[] hist = new double[T];
        final int nPix = H * W;
        for (int p = 0; p < nPix; p++) {
            final int base = p * T;
            for (int t = 0; t < T; t++) {
                final double v = Y[base + t];
                if (Double.isFinite(v) && v > 0.0) hist[t] += v;
            }
        }

        double total = 0.0;
        for (double v : hist) total += v;
        if (!(total > 0.0)) return 0.0;

        // Very short stacks: fall back to the global maximum.
        if (T < 5) {
            int imax = 0;
            for (int t = 1; t < T; t++) if (hist[t] > hist[imax]) imax = t;
            return imax;
        }

        // Five-point binomial smoothing [1, 4, 6, 4, 1] / 16,
        // renormalized at the boundaries.
        final int[] off = {-2, -1, 0, 1, 2};
        final double[] wt = {1.0, 4.0, 6.0, 4.0, 1.0};
        final double[] smooth = new double[T];
        for (int t = 0; t < T; t++) {
            double s = 0.0, sw = 0.0;
            for (int j = 0; j < off.length; j++) {
                final int q = t + off[j];
                if (q < 0 || q >= T) continue;
                s += wt[j] * hist[q];
                sw += wt[j];
            }
            smooth[t] = s / sw;
        }

        // Central-difference derivative. The IRF center is well approximated by
        // the maximum positive slope of an IRF-convolved exponential decay.
        final double[] deriv = new double[T];
        int best = 1;
        double bestSlope = Double.NEGATIVE_INFINITY;
        for (int t = 1; t < T - 1; t++) {
            deriv[t] = 0.5 * (smooth[t + 1] - smooth[t - 1]);
            if (deriv[t] > bestSlope) {
                bestSlope = deriv[t];
                best = t;
            }
        }

        // If no positive leading edge is detectable, use the global maximum.
        if (!(bestSlope > 0.0)) {
            int imax = 0;
            for (int t = 1; t < T; t++) if (hist[t] > hist[imax]) imax = t;
            return imax;
        }

        // Parabolic interpolation around the derivative maximum for sub-bin t0.
        double delta = 0.0;
        if (best >= 2 && best <= T - 3) {
            final double ym = deriv[best - 1];
            final double y0 = deriv[best];
            final double yp = deriv[best + 1];
            final double denom = ym - 2.0 * y0 + yp;
            if (Math.abs(denom) > 1e-15) {
                delta = 0.5 * (ym - yp) / denom;
                if (delta < -0.5) delta = -0.5;
                if (delta > 0.5) delta = 0.5;
            }
        }
        return best + delta;
    }

    /** IRF-convolved, window-normalized exponential decay dictionary. */
    public static double[] decayBases(double[] tausNs, int nBins, double dtNs,
                                      double irfFwhmNs, double irfPeakBin) {
        double sig = irfFwhmNs / FWHM_TO_SIGMA;
        double t0 = irfPeakBin * dtNs;
        int K = tausNs.length;
        double[] D = new double[K * nBins];
        for (int k = 0; k < K; k++) {
            double lam = 1.0 / tausNs[k];
            double rowSum = 0.0;
            for (int t = 0; t < nBins; t++) {
                double tt = t * dtNs;
                double arg = (t0 + lam * sig * sig - tt) / (Math.sqrt(2.0) * sig);
                double ex = 0.5 * lam * lam * sig * sig - lam * (tt - t0);
                if (ex > 700) ex = 700; else if (ex < -700) ex = -700;
                double d = 0.5 * lam * Math.exp(ex) * Erf.erfc(arg);
                if (d < 0) d = 0;
                D[k * nBins + t] = d;
                rowSum += d;
            }
            if (rowSum > 0)
                for (int t = 0; t < nBins; t++) D[k * nBins + t] /= rowSum;
        }
        return D;
    }

    /** Lifetime dictionary nodes tauMin..tauMax inclusive in tauStep increments. */
    public static double[] tauNodes(double tauMin, double tauMax, double tauStep) {
        int k = (int) Math.round((tauMax - tauMin) / tauStep) + 1;
        double[] taus = new double[k];
        for (int i = 0; i < k; i++) taus[i] = tauMin + i * tauStep;
        return taus;
    }
}
