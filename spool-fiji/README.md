# SPOOL for Fiji

## Installation and use

Enable the **SPOOL** update site in **Help > Update > Manage update sites**,
apply the update and restart Fiji. Open a photon-count stack with one slice per
time bin, then choose **Plugins > SPOOL > SPOOL Reconstruct (FLIM)**.

To test a development JAR, close Fiji, back up the existing SPOOL JAR outside
Fiji, and place the new `spool_fiji-*.jar` in `Fiji.app/plugins/`. Keep only one
SPOOL plugin JAR in that directory, then restart Fiji. A development build is
not published to the update site until the release workflow runs.

## GPU option

**Use GPU (auto-detect)** is off by default, preserving the existing CPU behavior.
When enabled, SPOOL:

1. Finds available OpenCL GPU devices with a working compiler.
2. Prefers a discrete GPU, then larger reported device memory. This heuristic
   does not guarantee selection of the fastest GPU.
3. Checks estimated memory requirements and compares a small reconstruction
   against the CPU solver before using a device.
4. Keeps the photon cube, dictionary and intermediate buffers on the GPU
   throughout all iterations.
5. Logs the device and precision. If detection, compilation, device allocation,
   or execution fails, it logs the reason and restarts from the original input
   using the CPU solver. OpenCL CPU devices are not reported as GPUs.

The plugin bundles the JOCL Java/native bridge. Python, PyTorch, CUDA Toolkit
and CLIJ2 are not required. A working vendor **OpenCL driver** is required for
GPU execution. The plugin does not install system drivers. NVIDIA, AMD and Intel
GPU support depends on the installed driver, operating system and available
JOCL native binary. Unsupported systems retain the Java CPU path. In particular,
do not assume Apple Silicon compatibility without testing that Fiji/JVM and
native-library combination.

Open **Window > Log** to confirm the actual backend. The final timing includes
device setup, kernel compilation, the numerical self-test and data transfers.
No speedup factor is promised; small images can take longer on a GPU.

## Numerical behavior

The reference CPU implementation is unchanged and uses double precision. The
GPU implementation uses single precision. It retains the existing discrete
pixel-integrated PSF, zero padding, centered crop, adjoint correlation,
dictionary sums, background, initialization, epsilon, damping exponent and
`1e-8` amplitude floor. The same lifetime and intensity output code is used.
Results are expected to agree within numerical tolerance, not bit for bit.
No relaxed-math compiler option is enabled.

The GPU needs approximately
`4 * (2*H*W*T + 3*K*H*W + H*W + K*T + PSF_height*PSF_width + K)` bytes for
buffers, plus driver overhead. SPOOL limits the estimate to 60% of reported
device memory and checks the device's maximum single allocation. This is a
capacity check, not a measurement of currently free memory; an actual allocation
failure also triggers CPU fallback. Very large stacks still require adequate
Java heap memory for the existing CPU representation.

## Build and validation

```sh
mvn -f spool-fiji/pom.xml clean verify
```

The deployable artifact is `spool-fiji/target/spool_fiji-0.2.0.jar`.
The `original-*.jar` produced by the packaging step does not include JOCL and
must not be distributed as the GPU-enabled plugin.

The default tests check CPU selection, native-library absence, runtime failure,
invalid output, resource cleanup and input validation. OpenCL numerical tests
are opt-in locally and mandatory in CI:

```sh
mvn -f spool-fiji/pom.xml clean verify -Dspool.test.opencl=true
```

The OpenCL tests fail if explicitly enabled but no usable runtime is present.
CI uses POCL to execute the actual kernels on a CPU; this verifies the algorithm
and OpenCL API calls, but does not validate graphics hardware or measure GPU
speed. The fixtures include sparse photons, boundary emitters, asymmetric PSFs,
unnormalized dictionary rows, zero counts, small images, and 50 iterations.
The sparse-photon lifetime comparison requires a maximum difference below
`1e-4 ns` for that fixture; this is not a general scientific accuracy guarantee.

For graphics-card validation, run on a machine with a supported GPU:

```sh
mvn -f spool-fiji/pom.xml clean verify -Dspool.test.opencl=true -Dspool.test.requireGpu=true
```

Also run a representative experimental stack in Fiji with the checkbox both
off and on. Compare lifetime/intensity maps, record total time and device/driver
information, and check the log confirms GPU execution. The existing update-site
release workflow publishes only after its tests pass.

Version 0.2.0 has passed the POCL numerical tests. Physical GPU compatibility and
speed measurements have not yet been established. GPU use remains opt-in, with
the numerical self-test and CPU fallback enabled.

## Third-party component

JOCL is distributed under the MIT license. Its notice is included in the JAR
at `META-INF/SPOOL-THIRD-PARTY.txt`.
