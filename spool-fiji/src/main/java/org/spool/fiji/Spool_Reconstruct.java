package org.spool.fiji;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import org.spool.core.SpoolCore;
import org.spool.core.SpoolModel;

/** Fiji command for SPOOL FLIM reconstruction. */
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

    @Parameter(label = "Background (counts / pixel / bin)", min = "0")
    private double background = 0.0;

    @Parameter(label = "Iterations", min = "1")
    private int iterations = 50;

    @Parameter(label = "Damping exponent eta", min = "0.1", max = "1.0")
    private double eta = 0.9;

    @Parameter(label = "Amplitude mask threshold (fraction of max)", min = "0")
    private double maskFrac = 0.05;

    @Parameter(label = "Show component amplitude maps")
    private boolean showAmplitudes = false;

    private static final double EPS = 1e-9;

    @Override
    public void run() {
        final int W = imp.getWidth();
        final int H = imp.getHeight();
        final int T = imp.getStackSize();
        if (T < 8) {
            IJ.error("SPOOL", "Expected an x-y-t photon-count stack (one slice per time bin).");
            return;
        }

        IJ.showStatus("SPOOL: reading photon counts");
        final double[] Y = new double[H * W * T];
        final ImageStack st = imp.getStack();
        for (int t = 0; t < T; t++) {
            final float[] px = (float[]) st.getProcessor(t + 1).convertToFloatProcessor().getPixels();
            for (int p = 0; p < H * W; p++) Y[p * T + t] = px[p];
        }

        final double irfPeakBin = SpoolModel.estimateIrfPeakBin(Y, H, W, T);
        IJ.log(String.format(
                "SPOOL: auto-detected t0 = %.3f bins (%.4f ns)",
                irfPeakBin, irfPeakBin * dtNs));

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
        final double[] A = SpoolCore.joint(Y, D, psf, H, W, T, K,
                psfSize[0], psfSize[1], background, iterations, eta, EPS);
        final double secs = (System.currentTimeMillis() - t0) / 1000.0;

        final double[] tau = SpoolCore.parameterMap(A, taus, K, H, W, EPS);
        final double[] abundance = new double[H * W];
        double aMax = 0.0;
        for (int p = 0; p < H * W; p++) {
            double s = 0.0;
            for (int k = 0; k < K; k++) s += A[k * H * W + p];
            abundance[p] = s;
            if (s > aMax) aMax = s;
        }
        final float[] tauPx = new float[H * W];
        for (int p = 0; p < H * W; p++)
            tauPx[p] = abundance[p] >= maskFrac * aMax ? (float) tau[p] : Float.NaN;

        final ImagePlus tauImp = new ImagePlus(
                imp.getShortTitle() + " SPOOL lifetime (ns)",
                new FloatProcessor(W, H, tauPx));
        tauImp.show();
        IJ.run(tauImp, "Rainbow RGB", "");

        if (showAmplitudes) {
            final ImageStack amps = new ImageStack(W, H);
            for (int k = 0; k < K; k++) {
                final float[] ap = new float[H * W];
                for (int p = 0; p < H * W; p++) ap[p] = (float) A[k * H * W + p];
                amps.addSlice(String.format("tau = %.1f ns", taus[k]),
                        new FloatProcessor(W, H, ap));
            }
            new ImagePlus(imp.getShortTitle() + " SPOOL amplitudes", amps).show();
        }

        IJ.showStatus(String.format("SPOOL: done in %.1f s (t0=%.2f bins)", secs, irfPeakBin));
    }
}
