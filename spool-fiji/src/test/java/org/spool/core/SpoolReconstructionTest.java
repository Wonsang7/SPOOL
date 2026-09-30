package org.spool.core;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import static org.junit.Assert.*;
import org.spool.core.SpoolReconstruction.*;

public class SpoolReconstructionTest {
    private Request request() {
        return new Request(new double[]{0,1,3,2,2,1,0,1},
                new double[]{0.7,0.3,0.2,0.8}, new double[]{1},
                2,2,2,2,1,1,0.01,6,0.9,1e-9);
    }

    @Test public void disabledGpuNeverLoadsBackend() {
        Request r = request();
        Result result = SpoolReconstruction.run(r, false, s -> {}, (ignored, log) -> {
            throw new AssertionError("GPU factory must not be invoked");
        });
        assertArrayEquals(r.cpu(), result.amplitudes, 0);
        assertTrue(result.backend.startsWith("CPU"));
    }

    @Test public void missingNativeDriverFallsBackWithReason() {
        Request r = request();
        List<String> logs = new ArrayList<>();
        Result result = SpoolReconstruction.run(r, true, logs::add, (ignored, log) -> {
            throw new UnsatisfiedLinkError("OpenCL driver missing");
        });
        assertArrayEquals(r.cpu(), result.amplitudes, 0);
        assertTrue(logs.stream().anyMatch(s -> s.contains("OpenCL driver missing")));
    }

    @Test public void failedExecutionClosesBackendAndRestartsCpu() {
        Request r = request();
        double[] original = r.counts.clone();
        boolean[] closed = {false};
        Result result = SpoolReconstruction.run(r, true, s -> {}, (ignored, log) -> new Backend() {
            @Override public double[] reconstruct(Request request) {
                throw new IllegalStateException("CL_MEM_OBJECT_ALLOCATION_FAILURE");
            }
            @Override public String description() { return "test GPU"; }
            @Override public void close() { closed[0] = true; }
        });
        assertTrue(closed[0]);
        assertArrayEquals(original, r.counts, 0);
        assertArrayEquals(r.cpu(), result.amplitudes, 0);
    }

    @Test public void nonFiniteGpuOutputIsRejected() {
        Request r = request();
        Result result = SpoolReconstruction.run(r, true, s -> {}, (ignored, log) -> new Backend() {
            @Override public double[] reconstruct(Request request) {
                double[] values = request.cpu();
                values[0] = Double.NaN;
                return values;
            }
            @Override public String description() { return "test GPU"; }
            @Override public void close() {}
        });
        assertTrue(result.backend.startsWith("CPU"));
        assertArrayEquals(r.cpu(), result.amplitudes, 0);
    }

    @Test public void successfulBackendIsReportedAndClosed() {
        Request r = request();
        boolean[] closed = {false};
        Result result = SpoolReconstruction.run(r, true, s -> {}, (ignored, log) -> new Backend() {
            @Override public double[] reconstruct(Request request) { return request.cpu(); }
            @Override public String description() { return "test GPU"; }
            @Override public void close() { closed[0] = true; }
        });
        assertTrue(closed[0]);
        assertEquals("test GPU", result.backend);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonfinitePhotonCounts() {
        new Request(new double[]{Double.NaN}, new double[]{1}, new double[]{1},
                1,1,1,1,1,1,0,1,0.9,1e-9);
    }

    @Test public void cpuStillRunsWhenJoclIsNotOnClasspath() throws Exception {
        String javaExecutable = System.getProperty("java.home") + "/bin/java";
        String classpath = SpoolCore.class.getProtectionDomain().getCodeSource().getLocation().getPath() +
                java.io.File.pathSeparator + NoJoclSmoke.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        Process process = new ProcessBuilder(javaExecutable, "-cp", classpath, NoJoclSmoke.class.getName())
                .inheritIO().start();
        assertEquals(0, process.waitFor());
    }

    /** Separate JVM intentionally excludes JOCL and all other dependencies. */
    public static class NoJoclSmoke {
        public static void main(String[] args) {
            Request r = new SpoolReconstructionTest().request();
            for (boolean gpu : new boolean[]{false, true}) {
                Result result = SpoolReconstruction.run(r, gpu, System.out::println);
                if (!result.backend.startsWith("CPU") || result.amplitudes.length != 8)
                    throw new AssertionError("CPU fallback failed without JOCL");
            }
        }
    }
}
