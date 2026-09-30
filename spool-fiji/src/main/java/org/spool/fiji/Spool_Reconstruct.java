package org.spool.fiji;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;

import java.awt.Color;

import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import org.spool.core.SpoolCore;
import org.spool.core.SpoolModel;
import org.spool.core.SpoolReconstruction;

/**
 * SPOOL: photon-efficient FLIM reconstruction (training-free Poisson inverse
 * with an explicit PSF forward operator).
 *
 * Input : an x-y-t photon-count stack, one slice per TCSPC time bin.
 * Output: SPOOL intensity, lifetime, and intensity-weighted lifetime maps.
 */
@Plugin(type = Command.class, menuPath = "Plugins>SPOOL>SPOOL Reconstruct (FLIM)")
public class Spool_Reconstruct implements Command {

    @Parameter(label = "Photon-count stack (x, y, t)")
    private ImagePlus imp;

    @Parameter(label = "PSF FWHM (nm)", min = "50")
    private double psfFwhmNm = 200.0;

    @Parameter(label = "Pixel size (nm)", min = "1")
    private double pixelNm = 50.0;

    @Parameter(label = "Time-bin width dt (ns)", min = "0")
    private double dtNs = 0.0977;

    @Parameter(label = "IRF FWHM (ns)", min = "0")
    private double irfFwhmNs = 0.150;

    @Parameter(label = "Dictionary tau min (ns)")
    private double tauMin = 1.0;

    @Parameter(label = "Dictionary tau max (ns)")
    private double tauMax = 5.0;

    @Parameter(label = "Dictionary tau step (ns)")
    private double tauStep = 0.4;

    @Parameter(label = "Auto-estimate background from darkest pixels",
               description = "Mean counts per bin over the dimmest 10% of pixels. Disable to enter background manually.")
    private boolean autoBackground = true;

    @Parameter(label = "Background (counts / pixel / bin, if not auto)", min = "0")
    private double background = 0.0;

    @Parameter(label = "Iterations", min = "1")
    private int iterations = 50;

    @Parameter(label = "Use GPU (auto-detect)",
               description = "Use a compatible OpenCL GPU when available. Otherwise restart on CPU and report the reason in the Log window.")
    private boolean useGpu = false;

    @Parameter(label = "Damping exponent eta", min = "0.1", max = "1.0")
    private double eta = 0.9;

    @Parameter(label = "Intensity mask threshold (fraction of max)", min = "0")
    private double maskFrac = 0.05;

    private static final double EPS = 1e-9;

    @Override
    public void run() {
        if (imp == null) {
            IJ.error("SPOOL", "Please open a photon-count stack first.");
            return;
        }
        if (!Double.isFinite(psfFwhmNm) || psfFwhmNm <= 0 ||
                !Double.isFinite(pixelNm) || pixelNm <= 0 ||
                !Double.isFinite(dtNs) || dtNs <= 0 ||
                !Double.isFinite(irfFwhmNs) || irfFwhmNs <= 0 ||
                !Double.isFinite(tauMin) || tauMin <= 0 ||
                !Double.isFinite(tauMax) || tauMax < tauMin ||
                !Double.isFinite(tauStep) || tauStep <= 0 ||
                !Double.isFinite(background) || background < 0 ||
                iterations < 1 || !Double.isFinite(eta) || eta <= 0 || eta > 1 ||
                !Double.isFinite(maskFrac) || maskFrac < 0) {
            IJ.error("SPOOL", "Please enter finite, valid reconstruction parameters. Widths and lifetimes must be positive.");
            return;
        }
        final int W = imp.getWidth();
        final int H = imp.getHeight();
        final int T = imp.getStackSize();
        if (T < 8) {
            IJ.error("SPOOL", "Expected an x-y-t photon-count stack (one slice per time bin).");
            return;
        }

        final double componentCount = Math.round((tauMax - tauMin) / tauStep) + 1.0;
        final double psfSide = 2.0 * Math.max(Math.ceil(4.0 *
                (psfFwhmNm / SpoolModel.FWHM_TO_SIGMA) / pixelNm), 1.0) + 1.0;
        if ((long) H * W * T > Integer.MAX_VALUE ||
                componentCount * H * W > Integer.MAX_VALUE ||
                componentCount * T > Integer.MAX_VALUE || psfSide * psfSide > Integer.MAX_VALUE) {
            IJ.error("SPOOL", "The stack or model exceeds Java array limits. Use a smaller stack or dictionary.");
            return;
        }

        // Gather the cube as (h*W + w)*T + t.
        IJ.showStatus("SPOOL: reading photon counts");
        final double[] Y = new double[H * W * T];
        final ImageStack st = imp.getStack();
        for (int t = 0; t < T; t++) {
            final float[] px = (float[]) st.getProcessor(t + 1).convertToFloatProcessor().getPixels();
            for (int p = 0; p < H * W; p++) {
                if (!Float.isFinite(px[p]) || px[p] < 0) {
                    IJ.error("SPOOL", "Photon counts must be finite and nonnegative.");
                    return;
                }
                Y[p * T + t] = px[p];
            }
        }

        // Automatic t0 detection from the global leading edge. This avoids the
        // lifetime-dependent bias of simply using the fluorescence maximum.
        final double irfPeakBin = SpoolModel.estimateIrfPeakBin(Y, H, W, T);
        IJ.log(String.format(
                "SPOOL: auto-detected t0 = %.3f bins (%.4f ns)",
                irfPeakBin, irfPeakBin * dtNs));

        // Optional background calibration from the dimmest 10% of pixels.
        if (autoBackground) {
            final double[] rawIntensity = new double[H * W];
            for (int p = 0; p < H * W; p++) {
                double s = 0.0;
                for (int t = 0; t < T; t++) s += Y[p * T + t];
                rawIntensity[p] = s;
            }
            final double[] sorted = rawIntensity.clone();
            java.util.Arrays.sort(sorted);
            final int cutIndex = Math.max(0, (int) Math.ceil(0.10 * sorted.length) - 1);
            final double cut = sorted[cutIndex];
            double bgSum = 0.0;
            long n = 0;
            for (int p = 0; p < H * W; p++) {
                if (rawIntensity[p] <= cut) {
                    bgSum += rawIntensity[p];
                    n++;
                }
            }
            background = n > 0 ? bgSum / (n * (double) T) : 0.0;
            IJ.log(String.format(
                    "SPOOL: auto-estimated background = %.4g counts/pixel/bin (%d pixels)",
                    background, n));
        }

        // Forward model.
        final double sigmaPx = (psfFwhmNm / SpoolModel.FWHM_TO_SIGMA) / pixelNm;
        final int[] psfSize = new int[2];
        final double[] psf = SpoolModel.gaussianPsf(sigmaPx, psfSize);
        final double[] taus = SpoolModel.tauNodes(tauMin, tauMax, tauStep);
        final double[] D = SpoolModel.decayBases(taus, T, dtNs, irfFwhmNs, irfPeakBin);
        final int K = taus.length;

        IJ.showStatus(String.format(
                "SPOOL: reconstructing (t0=%.2f bins, K=%d, %d iterations)",
                irfPeakBin, K, iterations));
        final long t0 = System.currentTimeMillis();
        final SpoolReconstruction.Result reconstruction;
        try {
            final SpoolReconstruction.Request request = new SpoolReconstruction.Request(
                    Y, D, psf, H, W, T, K, psfSize[0], psfSize[1],
                    background, iterations, eta, EPS);
            reconstruction = SpoolReconstruction.run(request, useGpu, message -> {
                IJ.log(message);
                IJ.showStatus(message);
            });
        } catch (IllegalArgumentException failure) {
            IJ.error("SPOOL", failure.getMessage());
            return;
        }
        final double[] A = reconstruction.amplitudes;
        final double secs = (System.currentTimeMillis() - t0) / 1000.0;

        // Lifetime map and reconstructed source-space intensity.
        final double[] tau = SpoolCore.parameterMap(A, taus, K, H, W, EPS);
        final double[] sourceIntensity = new double[H * W];
        double iMax = 0.0;
        for (int p = 0; p < H * W; p++) {
            double s = 0.0;
            for (int k = 0; k < K; k++) s += A[k * H * W + p];
            sourceIntensity[p] = s;
            if (s > iMax) iMax = s;
        }

        final float[] tauPx = new float[H * W];
        final float[] intPx = new float[H * W];
        for (int p = 0; p < H * W; p++) {
            tauPx[p] = sourceIntensity[p] >= maskFrac * iMax ? (float) tau[p] : Float.NaN;
            intPx[p] = (float) sourceIntensity[p];
        }

        // Output 1: source-space SPOOL intensity, displayed with Red Hot LUT.
        final ImagePlus intImp = new ImagePlus(
                imp.getShortTitle() + " SPOOL intensity",
                new FloatProcessor(W, H, intPx));
        intImp.show();
        IJ.run(intImp, "Red Hot", "");

        // Shared lifetime color scale for the lifetime and weighted-lifetime outputs.
        final java.util.ArrayList<Float> vals = new java.util.ArrayList<>();
        for (int p = 0; p < H * W; p++) {
            if (!Float.isNaN(tauPx[p])) vals.add(tauPx[p]);
        }
        float tLo = (float) tauMin;
        float tHi = (float) tauMax;
        if (vals.size() > 10) {
            java.util.Collections.sort(vals);
            tLo = vals.get((int) (0.02 * (vals.size() - 1)));
            tHi = vals.get((int) (0.98 * (vals.size() - 1)));
            if (tHi - tLo < 1e-3f) {
                tLo -= 0.05f;
                tHi += 0.05f;
            }
        }

        final byte[] rL = new byte[256];
        final byte[] gL = new byte[256];
        final byte[] bL = new byte[256];
        for (int i = 0; i < 256; i++) {
            final int c = Color.HSBtoRGB(0.75f * (1f - i / 255f), 1f, 1f);
            rL[i] = (byte) ((c >> 16) & 0xff);
            gL[i] = (byte) ((c >> 8) & 0xff);
            bL[i] = (byte) (c & 0xff);
        }
        final ij.process.LUT sharedLut = new ij.process.LUT(rL, gL, bL);
        IJ.log(String.format(
                "SPOOL: lifetime color scale = %.2f-%.2f ns", tLo, tHi));

        // Output 2: intensity-masked lifetime map.
        final ImagePlus tauImp = new ImagePlus(
                imp.getShortTitle() + " SPOOL lifetime (ns)",
                new FloatProcessor(W, H, tauPx));
        tauImp.setLut(sharedLut);
        tauImp.setDisplayRange(tLo, tHi);
        tauImp.show();

        // Output 3: lifetime as hue and recovered SPOOL intensity as brightness.
        final double[] sortedIntensity = sourceIntensity.clone();
        java.util.Arrays.sort(sortedIntensity);
        final double iSat = Math.max(
                sortedIntensity[(int) (0.995 * (sortedIntensity.length - 1))], 1e-12);

        final int[] rgb = new int[H * W];
        for (int p = 0; p < H * W; p++) {
            final float brightness = (float) Math.min(sourceIntensity[p] / iSat, 1.0);
            final float frac = Float.isNaN(tauPx[p]) ? 0f
                    : Math.min(Math.max((tauPx[p] - tLo) / (tHi - tLo), 0f), 1f);
            final float hue = 0.75f * (1f - frac); // short tau blue/violet, long tau red
            rgb[p] = Color.HSBtoRGB(hue, 1f, brightness);
        }
        final ImagePlus weightedImp = new ImagePlus(
                imp.getShortTitle() + " SPOOL intensity-weighted lifetime",
                new ColorProcessor(W, H, rgb));
        weightedImp.show();

        IJ.showStatus(String.format(
                "SPOOL: done in %.1f s using %s (t0=%.2f bins, bg=%.4g)",
                secs, reconstruction.backend, irfPeakBin, background));
        IJ.log(String.format("SPOOL: completed in %.3f s using %s", secs, reconstruction.backend));
    }
}
