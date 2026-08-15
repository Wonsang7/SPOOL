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
