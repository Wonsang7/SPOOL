// SPOOL FP32 kernels. Deliberately no fast-relaxed-math.
// Y/R are time-major (t*N+p); A/Ac/U are component-major (k*N+p).
// Explicit bounds preserve the CPU solver's zero-padded, centered same crop.

__kernel void convolve(__global const float *a, __global const float *psf,
        __global float *ac, int h, int w, int ph, int pw) {
    int p = get_global_id(0), k = get_global_id(1), n = h*w;
    int y = p/w, x = p%w;
    float sum = 0.0f;
    for (int u = 0; u < ph; ++u) {
        int yy = y-u+(ph-1)/2;
        if (yy < 0 || yy >= h) continue;
        for (int v = 0; v < pw; ++v) {
            int xx = x-v+(pw-1)/2;
            if (xx >= 0 && xx < w)
                sum += psf[u*pw+v] * a[k*n+yy*w+xx];
        }
    }
    ac[k*n+p] = sum;
}

__kernel void ratios(__global const float *y, __global const float *d,
        __global const float *ac, __global float *r,
        int n, int tCount, int kCount, float bg, float eps) {
    int p = get_global_id(0), t = get_global_id(1);
    float lambda = bg;
    for (int k = 0; k < kCount; ++k)
        lambda += ac[k*n+p] * d[k*tCount+t];
    r[t*n+p] = y[t*n+p] / (lambda + eps);
}

__kernel void project(__global const float *d, __global const float *r,
        __global float *u, int n, int tCount) {
    int p = get_global_id(0), k = get_global_id(1);
    float sum = 0.0f;
    for (int t = 0; t < tCount; ++t)
        sum += d[k*tCount+t] * r[t*n+p];
    u[k*n+p] = sum;
}

__kernel void update(__global float *a, __global const float *u,
        __global const float *psf, __global const float *convOnes,
        __global const float *dSum, int h, int w, int ph, int pw,
        float eta, float eps) {
    int p = get_global_id(0), k = get_global_id(1), n = h*w;
    int y = p/w, x = p%w;
    float sum = 0.0f;
    for (int i = 0; i < ph; ++i) {
        int yy = y+i-(ph-1)/2;
        if (yy < 0 || yy >= h) continue;
        for (int j = 0; j < pw; ++j) {
            int xx = x+j-(pw-1)/2;
            if (xx >= 0 && xx < w)
                sum += psf[i*pw+j] * u[k*n+yy*w+xx];
        }
    }
    float ratio = sum / (convOnes[p] * dSum[k] + eps);
    // Preserve NaN/Inf so the host detects failures instead of hiding them.
    if (ratio < 0.0f) ratio = 0.0f;
    float next = a[k*n+p] * pow(ratio, eta);
    a[k*n+p] = next < 1.0e-8f ? 1.0e-8f : next;
}
