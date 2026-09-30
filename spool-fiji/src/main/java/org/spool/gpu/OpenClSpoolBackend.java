package org.spool.gpu;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Consumer;

import org.jocl.*;
import static org.jocl.CL.*;
import org.spool.core.SpoolCore;
import org.spool.core.SpoolReconstruction;
import org.spool.core.SpoolReconstruction.Request;

/** Optional OpenCL 1.2 backend. All iteration buffers remain on the device. */
public final class OpenClSpoolBackend implements SpoolReconstruction.Backend {
    private final Device device;
    private cl_context context;
    private cl_command_queue queue;
    private cl_program program;
    private final List<cl_kernel> kernels = new ArrayList<>();
    private cl_kernel convolve, ratios, project, update;

    private static final class Device {
        final cl_platform_id platform;
        final cl_device_id id;
        final String name;
        final long memory, maxAllocation;
        final boolean unifiedMemory;

        Device(cl_platform_id platform, cl_device_id id) {
            this.platform = platform;
            this.id = id;
            name = deviceString(id, CL_DEVICE_NAME).trim();
            memory = deviceLong(id, CL_DEVICE_GLOBAL_MEM_SIZE);
            maxAllocation = deviceLong(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            unifiedMemory = deviceInt(id, CL_DEVICE_HOST_UNIFIED_MEMORY) != 0;
        }
    }

    /** Production selection deliberately excludes OpenCL CPU devices. */
    public static SpoolReconstruction.Backend openGpu(Request request, Consumer<String> log) {
        return open(request, log, CL_DEVICE_TYPE_GPU);
    }

    // Package-private test entry point allows POCL CPU execution of the identical
    // kernels in CI. It cannot be selected from the Fiji interface.
    static OpenClSpoolBackend open(Request request, Consumer<String> log, long deviceType) {
        List<Device> devices = devices(deviceType, log);
        // Prefer discrete GPUs, then the largest device memory. This is a
        // deterministic heuristic, not a claim to select the fastest device.
        devices.sort(Comparator.comparing((Device d) -> d.unifiedMemory)
                .thenComparing(Comparator.comparingLong((Device d) -> d.memory).reversed()));
        for (Device device : devices) {
            OpenClSpoolBackend backend = null;
            try {
                checkMemory(request, device.memory, device.maxAllocation);
                backend = new OpenClSpoolBackend(device);
                backend.selfTest();
                return backend;
            } catch (RuntimeException | LinkageError failure) {
                if (backend != null) backend.close();
                log.accept("SPOOL: skipping " + device.name + " (" +
                        SpoolReconstruction.reason(failure) + ")");
            }
        }
        throw new IllegalStateException("No compatible OpenCL GPU passed the memory and numerical checks. " +
                "A vendor OpenCL driver is required; see the Fiji GPU instructions.");
    }

    private static List<Device> devices(long type, Consumer<String> log) {
        int[] count = new int[1];
        check(clGetPlatformIDs(0, null, count), "enumerate OpenCL platforms");
        if (count[0] == 0) throw new IllegalStateException("No OpenCL platforms found");
        cl_platform_id[] platforms = new cl_platform_id[count[0]];
        check(clGetPlatformIDs(platforms.length, platforms, null), "read OpenCL platforms");
        List<Device> result = new ArrayList<>();
        for (cl_platform_id platform : platforms) {
            try {
                count[0] = 0;
                int status = clGetDeviceIDs(platform, type, 0, null, count);
                if (status == CL_DEVICE_NOT_FOUND || count[0] == 0) continue;
                check(status, "enumerate OpenCL devices");
                cl_device_id[] ids = new cl_device_id[count[0]];
                check(clGetDeviceIDs(platform, type, ids.length, ids, null), "read OpenCL devices");
                for (cl_device_id id : ids) {
                    try {
                        if (deviceInt(id, CL_DEVICE_AVAILABLE) != 0 &&
                                deviceInt(id, CL_DEVICE_COMPILER_AVAILABLE) != 0) {
                            result.add(new Device(platform, id));
                        }
                    } catch (RuntimeException failure) {
                        log.accept("SPOOL: cannot inspect an OpenCL device: " + failure.getMessage());
                    }
                }
            } catch (RuntimeException failure) {
                log.accept("SPOOL: cannot use an OpenCL platform: " + failure.getMessage());
            }
        }
        return result;
    }

    @SuppressWarnings("deprecation")
    private OpenClSpoolBackend(Device device) {
        this.device = device;
        try {
            cl_context_properties properties = new cl_context_properties();
            properties.addProperty(CL_CONTEXT_PLATFORM, device.platform);
            int[] error = new int[1];
            context = clCreateContext(properties, 1, new cl_device_id[]{device.id}, null, null, error);
            check(error[0], "create OpenCL context");
            // In-order OpenCL 1.2 queue for broad driver compatibility.
            queue = clCreateCommandQueue(context, device.id, 0, error);
            check(error[0], "create OpenCL queue");
            program = clCreateProgramWithSource(context, 1, new String[]{source()}, null, error);
            check(error[0], "create SPOOL OpenCL program");
            int status;
            try {
                status = clBuildProgram(program, 1, new cl_device_id[]{device.id},
                        "-cl-std=CL1.2", null, null);
            } catch (RuntimeException failure) {
                throw new IllegalStateException("SPOOL OpenCL compilation failed: " + buildLog(), failure);
            }
            if (status != CL_SUCCESS) {
                throw new IllegalStateException("SPOOL OpenCL compilation failed: " + buildLog());
            }
            convolve = kernel("convolve");
            ratios = kernel("ratios");
            project = kernel("project");
            update = kernel("update");
        } catch (RuntimeException | LinkageError failure) {
            close();
            throw failure;
        }
    }

    @Override public String description() {
        return "OpenCL: " + device.name + " (single precision)";
    }

    /** Conservative capacity estimate; actual allocation can still fail safely. */
    static void checkMemory(Request r, long globalBytes, long maxAllocationBytes) {
        long n = (long) r.height * r.width;
        long nt = n * r.bins, nk = n * r.components;
        long largest = Math.max(Math.max(nt, nk), Math.max(r.dictionary.length, r.psf.length));
        long total = (2*nt + 3*nk + n + r.dictionary.length + r.psf.length + r.components) * 4;
        if (largest * 4 > maxAllocationBytes || total > globalBytes * 0.60) {
            throw new IllegalArgumentException("Insufficient GPU memory for this stack (about " +
                    ((total + (1L << 20) - 1) >> 20) + " MiB of buffers required)");
        }
    }

    @Override public double[] reconstruct(Request r) {
        checkMemory(r, device.memory, device.maxAllocation);
        int n = r.height * r.width, nk = n * r.components;
        float bg = scalar(r.background), eta = scalar(r.eta), eps = scalar(r.eps);
        if (eps == 0 || eta == 0) throw new IllegalArgumentException("Parameters underflow in GPU precision");
        List<cl_mem> buffers = new ArrayList<>();
        try {
            cl_mem y = uploadCounts(r, buffers);
            cl_mem d = upload(r.dictionary, buffers);
            cl_mem psf = upload(r.psf, buffers);
            double[] ones = new double[n];
            Arrays.fill(ones, 1.0);
            // Keep the reference solver's exact boundary normalization.
            cl_mem norm = upload(SpoolCore.conv2dSame(ones, r.height, r.width,
                    r.psf, r.psfHeight, r.psfWidth), buffers);
            double[] dSum = new double[r.components];
            for (int k = 0; k < r.components; k++) {
                for (int t = 0; t < r.bins; t++) dSum[k] += r.dictionary[k*r.bins+t];
            }
            cl_mem ds = upload(dSum, buffers);
            cl_mem a = initialAmplitudes(r, nk, buffers);
            cl_mem ac = allocate(nk, buffers);
            cl_mem ratio = allocate(r.counts.length, buffers);
            cl_mem u = allocate(nk, buffers);

            args(convolve, a, psf, ac, r.height, r.width, r.psfHeight, r.psfWidth);
            args(ratios, y, d, ac, ratio, n, r.bins, r.components, bg, eps);
            args(project, d, ratio, u, n, r.bins);
            args(update, a, u, psf, norm, ds, r.height, r.width,
                    r.psfHeight, r.psfWidth, eta, eps);
            long[] maps = {n, r.components}, cube = {n, r.bins};
            for (int iteration = 0; iteration < r.iterations; iteration++) {
                enqueue(convolve, maps);
                enqueue(ratios, cube);
                enqueue(project, maps);
                enqueue(update, maps);
            }
            float[] output = new float[nk];
            check(clEnqueueReadBuffer(queue, a, CL_TRUE, 0, (long) nk * Sizeof.cl_float,
                    Pointer.to(output), 0, null, null), "read SPOOL result");
            double[] result = new double[nk];
            for (int i = 0; i < nk; i++) {
                if (!Float.isFinite(output[i]) || output[i] < 0) {
                    throw new IllegalStateException("GPU produced a non-finite or negative amplitude");
                }
                result[i] = output[i];
            }
            return result;
        } finally {
            for (cl_mem buffer : buffers) releaseQuietly(() -> clReleaseMemObject(buffer));
        }
    }

    /** Tiny end-to-end check before trusting a driver with the user's data. */
    private void selfTest() {
        double[] y = new double[5*4*8];
        for (int i = 0; i < y.length; i++) y[i] = i % 11 == 0 ? 3 : i % 3;
        double[] d = {0.8,0.6,0.4,0.2,0.1,0.05,0.02,0.01,
                      0.1,0.15,0.2,0.3,0.35,0.3,0.2,0.1};
        double[] psf = {0.01,0.02,0.03,0.04,0.6,0.1,0.06,0.06,0.08};
        Request r = new Request(y, d, psf, 5, 4, 8, 2, 3, 3, 0.03, 3, 0.9, 1e-9);
        double[] expected = r.cpu(), actual = reconstruct(r);
        for (int i = 0; i < expected.length; i++) {
            if (Math.abs(actual[i] - expected[i]) > 2e-5 + 2e-4*Math.abs(expected[i])) {
                throw new IllegalStateException("GPU numerical self-test did not match the CPU reference");
            }
        }
    }

    private cl_mem initialAmplitudes(Request r, int size, List<cl_mem> buffers) {
        double sum = 0;
        for (double value : r.counts) sum += value;
        float[] initial = new float[size];
        Arrays.fill(initial, scalar(Math.max(sum / size, 1e-4)));
        return upload(initial, buffers);
    }

    private cl_mem uploadCounts(Request r, List<cl_mem> buffers) {
        int n = r.height*r.width;
        float[] transposed = new float[r.counts.length];
        for (int p = 0; p < n; p++) {
            for (int t = 0; t < r.bins; t++) transposed[t*n+p] = scalar(r.counts[p*r.bins+t]);
        }
        return upload(transposed, buffers);
    }

    private cl_mem upload(double[] input, List<cl_mem> buffers) {
        float[] floats = new float[input.length];
        for (int i = 0; i < input.length; i++) floats[i] = scalar(input[i]);
        return upload(floats, buffers);
    }

    private cl_mem upload(float[] input, List<cl_mem> buffers) {
        return buffer(input.length, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Pointer.to(input), buffers);
    }

    private cl_mem allocate(int length, List<cl_mem> buffers) {
        return buffer(length, CL_MEM_READ_WRITE, null, buffers);
    }

    private cl_mem buffer(int length, long flags, Pointer data, List<cl_mem> buffers) {
        int[] error = new int[1];
        cl_mem memory = clCreateBuffer(context, flags, (long) length * Sizeof.cl_float, data, error);
        check(error[0], "allocate SPOOL GPU buffer");
        buffers.add(memory);
        return memory;
    }

    private cl_kernel kernel(String name) {
        int[] error = new int[1];
        cl_kernel kernel = clCreateKernel(program, name, error);
        check(error[0], "create kernel " + name);
        kernels.add(kernel);
        return kernel;
    }

    private static void args(cl_kernel kernel, Object... args) {
        for (int i = 0; i < args.length; i++) {
            Object value = args[i];
            if (value instanceof cl_mem) {
                check(clSetKernelArg(kernel, i, Sizeof.cl_mem, Pointer.to((cl_mem) value)), "set buffer argument");
            } else if (value instanceof Integer) {
                check(clSetKernelArg(kernel, i, Sizeof.cl_int, Pointer.to(new int[]{(Integer) value})), "set integer argument");
            } else if (value instanceof Float) {
                check(clSetKernelArg(kernel, i, Sizeof.cl_float, Pointer.to(new float[]{(Float) value})), "set float argument");
            } else throw new IllegalArgumentException("Unsupported OpenCL argument");
        }
    }

    private void enqueue(cl_kernel kernel, long[] size) {
        check(clEnqueueNDRangeKernel(queue, kernel, size.length, null, size, null,
                0, null, null), "execute SPOOL kernel");
    }

    private static float scalar(double value) {
        float result = (float) value;
        if (!Float.isFinite(result)) throw new IllegalArgumentException("Input exceeds GPU single precision range");
        return result;
    }

    private static void check(int status, String action) {
        if (status != CL_SUCCESS) throw new IllegalStateException(action + ": " + stringFor_errorCode(status));
    }

    private static long deviceLong(cl_device_id id, int field) {
        long[] value = new long[1];
        check(clGetDeviceInfo(id, field, Sizeof.cl_ulong, Pointer.to(value), null), "read device capacity");
        return value[0];
    }

    private static int deviceInt(cl_device_id id, int field) {
        int[] value = new int[1];
        check(clGetDeviceInfo(id, field, Sizeof.cl_uint, Pointer.to(value), null), "read device capability");
        return value[0];
    }

    private static String deviceString(cl_device_id id, int field) {
        long[] size = new long[1];
        check(clGetDeviceInfo(id, field, 0, null, size), "read device info length");
        byte[] bytes = new byte[(int) size[0]];
        check(clGetDeviceInfo(id, field, bytes.length, Pointer.to(bytes), null), "read device info");
        return new String(bytes, StandardCharsets.UTF_8).replace("\u0000", "");
    }

    private String buildLog() {
        long[] size = new long[1];
        clGetProgramBuildInfo(program, device.id, CL_PROGRAM_BUILD_LOG, 0, null, size);
        byte[] bytes = new byte[(int) size[0]];
        clGetProgramBuildInfo(program, device.id, CL_PROGRAM_BUILD_LOG, bytes.length, Pointer.to(bytes), null);
        return new String(bytes, StandardCharsets.UTF_8).replace("\u0000", "");
    }

    private static String source() {
        try (InputStream input = OpenClSpoolBackend.class.getResourceAsStream("spool.cl")) {
            if (input == null) throw new IllegalStateException("Missing SPOOL OpenCL kernel resource");
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] bytes = new byte[8192];
            int length;
            while ((length = input.read(bytes)) != -1) output.write(bytes, 0, length);
            return new String(output.toByteArray(), StandardCharsets.UTF_8);
        } catch (IOException failure) {
            throw new IllegalStateException("Cannot read SPOOL OpenCL kernels", failure);
        }
    }

    @Override public void close() {
        if (queue != null) releaseQuietly(() -> clFinish(queue));
        for (cl_kernel kernel : kernels) releaseQuietly(() -> clReleaseKernel(kernel));
        kernels.clear();
        if (program != null) releaseQuietly(() -> clReleaseProgram(program));
        if (queue != null) releaseQuietly(() -> clReleaseCommandQueue(queue));
        if (context != null) releaseQuietly(() -> clReleaseContext(context));
        program = null;
        queue = null;
        context = null;
    }

    private static void releaseQuietly(Runnable release) {
        try { release.run(); }
        catch (RuntimeException | LinkageError ignored) { /* continue releasing other resources */ }
    }
}
