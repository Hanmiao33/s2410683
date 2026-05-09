#include <vector>
#include <iostream>
#include <cmath>

static unsigned lcg_state;
static unsigned lcg_rand(){lcg_state=lcg_state*1103515245u+12345u;return lcg_state&0x7fffffffu;}
static float lcg_randf(){return (float)lcg_rand()/2147483648.0f;}

int main() {
    int n = 64;
    std::vector<float> A(n*n), b(n), x(n);
    lcg_state = 114514;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) A[i*n+j] = 0;
        A[i*n+i] = 1;
        for (int j = i+1; j < n; ++j) A[i*n+j] = lcg_randf();
    }
    for (int k = 0; k < n-1; ++k)
        for (int j = 0; j < n; ++j)
            A[(k+1)*n+j] += A[k*n+j];

    for (int i = 0; i < n; ++i) x[i] = (float)(i+1);
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int j = 0; j < n; ++j) s += (double)A[i*n+j] * x[j];
        b[i] = (float)s;
    }
    std::vector<float> Ao = A, bo = b;

    for (int k = 0; k < n; ++k) {
        float p = A[k*n+k];
        if (!std::isfinite(p)) { std::cout << "bad pivot k=" << k << std::endl; break; }
        for (int j = k+1; j < n; ++j) A[k*n+j] /= p;
        A[k*n+k] = 1; b[k] /= p;
        for (int i = k+1; i < n; ++i) {
            float f = A[i*n+k];
            for (int j = k+1; j < n; ++j) A[i*n+j] -= A[k*n+j] * f;
            b[i] -= b[k] * f; A[i*n+k] = 0;
        }
    }
    for (int i = n-1; i >= 0; --i) {
        float s = b[i];
        for (int j = i+1; j < n; ++j) s -= A[i*n+j] * b[j];
        b[i] = s / A[i*n+i];
    }

    double nr = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        double ax = 0;
        for (int j = 0; j < n; ++j) ax += (double)Ao[i*n+j] * b[j];
        double d = ax - bo[i];
        nr += d * d;
        nb += (double)bo[i] * bo[i];
    }
    std::cout << "norm_res=" << nr << " norm_b=" << nb
              << " sqrt_nr=" << sqrt(nr) << " sqrt_nb=" << sqrt(nb) << std::endl;
    std::cout << "x[0]=" << b[0] << " x[1]=" << b[1] << " x[2]=" << b[2]
              << " bo[0]=" << bo[0] << std::endl;
}
