package org.spool.gpu;

import java.util.Arrays;
import java.util.Random;
import org.junit.Assume;
import org.junit.Test;
import static org.junit.Assert.*;
import static org.jocl.CL.*;
import org.spool.core.SpoolCore;
import org.spool.core.SpoolModel;
import org.spool.core.SpoolReconstruction;
import org.spool.core.SpoolReconstruction.Request;

/** Enable with -Dspool.test.opencl=true; a missing runtime then FAILS the tests. */
public class OpenClSpoolBackendTest {
    private OpenClSpoolBackend backend(Request request) {
        Assume.assumeTrue("OpenCL integration tests explicitly enabled",
                Boolean.getBoolean("spool.test.opencl"));
        long type = Boolean.getBoolean("spool.test.requireGpu") ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL;
        return OpenClSpoolBackend.open(request, System.out::println, type);
    }

    @Test public void matchesCpuAtFiftyIterationsWithSparsePhotons() {
        int h = 13, w = 17, t = 64;
        double[] taus = SpoolModel.tauNodes(1,5,0.4);
        double[] d = SpoolModel.decayBases(taus,t,0.15,0.15,3.2);
        int[] shape = new int[2];
        double[] psf = SpoolModel.gaussianPsf(1.25,shape);
        double[] y = new double[h*w*t];
        Random random = new Random(239);
        for (int i = 0; i < y.length; i++) y[i] = random.nextDouble() < 0.015 ? 1 : 0;
        // Include both boundary and central emitters with higher counts.
        for (int i : new int[]{0, w-1, (h-1)*w, h*w-1, h*w/2})
            for (int bin = 0; bin < t; bin++) y[i*t+bin] += bin % 7 == 0 ? 2 : 0;
        Request r = new Request(y,d,psf,h,w,t,taus.length,shape[0],shape[1],1e-4,50,0.9,1e-9);
        try (OpenClSpoolBackend backend = backend(r)) {
            double[] original = y.clone(), cpu = r.cpu(), gpu = backend.reconstruct(r);
            assertRelative(cpu,gpu,3e-5,5e-4);
            assertArrayEquals(original,y,0);
            double[] cpuTau = SpoolCore.parameterMap(cpu,taus,taus.length,h,w,1e-9);
            double[] gpuTau = SpoolCore.parameterMap(gpu,taus,taus.length,h,w,1e-9);
            double maxTauError = 0;
            for (int p = 0; p < h*w; p++) maxTauError = Math.max(maxTauError,Math.abs(cpuTau[p]-gpuTau[p]));
            assertTrue("Maximum lifetime difference: " + maxTauError, maxTauError < 1e-4);
            System.out.println("50-iteration sparse-photon parity: max lifetime error = " + maxTauError + " ns; " + backend.description());
        }
    }

    @Test public void preservesAsymmetricPsfAndUnnormalizedDictionary() {
        int h=5,w=7,t=9,k=3;
        double[] y = new double[h*w*t], d = new double[k*t];
        Random random = new Random(89);
        for (int i=0;i<y.length;i++) y[i]=random.nextInt(4);
        for (int i=0;i<d.length;i++) d[i]=0.03+random.nextDouble();
        double[] psf={0.01,0.02,0.05,0.02,0.4,0.2,0.08,0.12,0.1};
        Request r=new Request(y,d,psf,h,w,t,k,3,3,0.2,30,0.7,1e-9);
        try (OpenClSpoolBackend backend=backend(r)) {
            assertRelative(r.cpu(),backend.reconstruct(r),2e-5,5e-4);
            // Reuse the same context to exercise per-run buffer cleanup.
            assertRelative(r.cpu(),backend.reconstruct(r),2e-5,5e-4);
        }
    }

    @Test public void handlesZeroCountsAndSingletonImage() {
        Request r=new Request(new double[16],new double[]{1,0.8,0.6,0.4,0.2,0.1,0.05,0.01,
                0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2},new double[]{0.1,0.2,0.1,0.1,0.2,0.1,0.05,0.1,0.05},
                1,2,8,2,3,3,0,50,0.9,1e-9);
        try (OpenClSpoolBackend backend=backend(r)) {
            assertRelative(r.cpu(),backend.reconstruct(r),1e-12,1e-5);
            Request single=new Request(new double[]{1,2,0,0,1,0,0,0},
                    Arrays.copyOf(r.dictionary,8),new double[]{1},1,1,8,1,1,1,0,4,1,1e-9);
            assertRelative(single.cpu(),backend.reconstruct(single),1e-6,1e-5);
        }
    }

    @Test public void productionSelectionNeverUsesOpenClCpu() {
        Assume.assumeTrue(Boolean.getBoolean("spool.test.cpuOnly"));
        Request r=new Request(new double[]{1},new double[]{1},new double[]{1},1,1,1,1,1,1,0,2,0.9,1e-9);
        SpoolReconstruction.Result result=SpoolReconstruction.run(r,true,System.out::println);
        assertTrue(result.backend.startsWith("CPU"));
        assertArrayEquals(r.cpu(),result.amplitudes,0);
    }

    @Test(expected=IllegalArgumentException.class)
    public void rejectsStacksExceedingSingleAllocationLimit() {
        Request r=new Request(new double[4*4*32],new double[2*32],new double[]{1},4,4,32,2,1,1,0,1,0.9,1e-9);
        OpenClSpoolBackend.checkMemory(r,1L<<30,1000);
    }

    private static void assertRelative(double[] expected,double[] actual,double absolute,double relative) {
        assertEquals(expected.length,actual.length);
        for (int i=0;i<expected.length;i++) {
            assertTrue("Non-finite GPU output at " + i,Double.isFinite(actual[i]));
            assertEquals("Amplitude at " + i,expected[i],actual[i],absolute+relative*Math.abs(expected[i]));
        }
    }
}
