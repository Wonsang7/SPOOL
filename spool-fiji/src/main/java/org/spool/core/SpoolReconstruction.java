package org.spool.core;

import java.util.function.Consumer;
import org.spool.gpu.OpenClSpoolBackend;

/** Selects the optional GPU backend without changing the reference CPU solver. */
public final class SpoolReconstruction {
    private SpoolReconstruction() {}

    public interface Backend extends AutoCloseable {
        double[] reconstruct(Request request);
        String description();
        @Override void close();
    }

    @FunctionalInterface
    public interface BackendFactory {
        Backend open(Request request, Consumer<String> log);
    }

    /** Input arrays are read-only for both backends, including after GPU failure. */
    public static final class Request {
        public final double[] counts, dictionary, psf;
        public final int height, width, bins, components, psfHeight, psfWidth, iterations;
        public final double background, eta, eps;

        public Request(double[] counts, double[] dictionary, double[] psf,
                int height, int width, int bins, int components, int psfHeight,
                int psfWidth, double background, int iterations, double eta, double eps) {
            if (height <= 0 || width <= 0 || bins <= 0 || components <= 0 ||
                    psfHeight <= 0 || psfWidth <= 0 || psfHeight % 2 == 0 || psfWidth % 2 == 0 ||
                    iterations < 1 || !Double.isFinite(background) || background < 0 ||
                    !Double.isFinite(eta) || eta <= 0 || eta > 1 ||
                    !Double.isFinite(eps) || eps <= 0) {
                throw new IllegalArgumentException("Invalid SPOOL dimensions or reconstruction parameters.");
            }
            final long pixels = (long) height * width;
            if (pixels > Integer.MAX_VALUE || pixels * bins > Integer.MAX_VALUE ||
                    pixels * components > Integer.MAX_VALUE ||
                    counts.length != pixels * bins || dictionary.length != (long) components * bins ||
                    psf.length != (long) psfHeight * psfWidth) {
                throw new IllegalArgumentException("SPOOL array dimensions do not match or exceed Java array limits.");
            }
            checkNonnegative(counts, "Photon counts");
            checkNonnegative(dictionary, "Dictionary");
            checkNonnegative(psf, "PSF");
            this.counts = counts;
            this.dictionary = dictionary;
            this.psf = psf;
            this.height = height;
            this.width = width;
            this.bins = bins;
            this.components = components;
            this.psfHeight = psfHeight;
            this.psfWidth = psfWidth;
            this.background = background;
            this.iterations = iterations;
            this.eta = eta;
            this.eps = eps;
        }

        private static void checkNonnegative(double[] values, String name) {
            for (double value : values) {
                if (!Double.isFinite(value) || value < 0) {
                    throw new IllegalArgumentException(name + " must be finite and nonnegative.");
                }
            }
        }

        public double[] cpu() {
            return SpoolCore.joint(counts, dictionary, psf, height, width, bins,
                    components, psfHeight, psfWidth, background, iterations, eta, eps);
        }
    }

    public static final class Result {
        public final double[] amplitudes;
        public final String backend;

        private Result(double[] amplitudes, String backend) {
            this.amplitudes = amplitudes;
            this.backend = backend;
        }
    }

    public static Result run(Request request, boolean useGpu, Consumer<String> log) {
        // The native bridge is initialized only when the checkbox is enabled.
        return run(request, useGpu, log, (r, logger) -> OpenClSpoolBackend.openGpu(r, logger));
    }

    public static Result run(Request request, boolean useGpu, Consumer<String> log,
            BackendFactory factory) {
        if (useGpu) {
            log.accept("SPOOL: searching for a compatible OpenCL GPU");
            try (Backend backend = factory.open(request, log)) {
                log.accept("SPOOL: using " + backend.description());
                final double[] amplitudes = backend.reconstruct(request);
                if (amplitudes == null || amplitudes.length !=
                        request.components * request.height * request.width) {
                    throw new IllegalStateException("GPU returned an invalid output size");
                }
                for (double value : amplitudes) {
                    if (!Double.isFinite(value) || value < 0) {
                        throw new IllegalStateException("GPU returned non-finite or negative amplitudes");
                    }
                }
                return new Result(amplitudes, backend.description());
            } catch (RuntimeException | LinkageError failure) {
                // Device absence, native library errors, build errors and device
                // allocation failures must not prevent the CPU implementation.
                // Do not swallow JVM OutOfMemoryError or other fatal VM errors.
                log.accept("SPOOL: GPU unavailable or failed (" + reason(failure) +
                        "). Restarting the reconstruction on CPU.");
            }
        }
        log.accept("SPOOL: using CPU (double precision)");
        return new Result(request.cpu(), "CPU (double precision)");
    }

    public static String reason(Throwable failure) {
        String message = failure.getMessage();
        return failure.getClass().getSimpleName() +
                (message == null ? "" : ": " + message);
    }
}
