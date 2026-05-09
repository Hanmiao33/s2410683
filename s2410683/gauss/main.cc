#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <arm_neon.h>
#include <cmath>

// 用自定义 LCG 替代 std::rand()，保证跨平台结果一致
static unsigned lcg_state;
static unsigned lcg_rand() {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return lcg_state & 0x7fffffffu;
}

static float lcg_randf() {
    return (float)lcg_rand() / 2147483648.0f;
}

void generate_test(float *A, float *b, float *x_true, int n, int seed)
{
    lcg_state = (unsigned)seed;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j)
            A[i * n + j] = 0.0f;
        A[i * n + i] = 1.0f;
        for (int j = i + 1; j < n; ++j)
            A[i * n + j] = lcg_randf();
    }
    for (int k = 0; k < n - 1; ++k)
        for (int j = 0; j < n; ++j)
            A[(k + 1) * n + j] += A[k * n + j];

    for (int i = 0; i < n; ++i)
        x_true[i] = (float)(i + 1);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            sum += (double)A[i * n + j] * (double)x_true[j];
        b[i] = (float)sum;
    }
}

bool load_from_file(const std::string &path, std::vector<float> &A,
                    std::vector<float> &b, int &n)
{
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    fin >> n;
    A.resize(n * n);
    b.resize(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            fin >> A[i * n + j];
        fin >> b[i];
    }
    fin.close();
    return true;
}

void save_solution(const std::string &path, const float *x, int n)
{
    std::ofstream fout(path);
    for (int i = 0; i < n; ++i)
        fout << std::fixed << std::setprecision(8) << x[i] << '\n';
}

float verify_solution(const float *A, const float *x, const float *b_orig, int n)
{
    double norm_res = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; ++i) {
        double ax = 0.0;
        for (int j = 0; j < n; ++j)
            ax += (double)A[i * n + j] * (double)x[j];
        double diff = ax - (double)b_orig[i];
        norm_res += diff * diff;
        norm_b += (double)b_orig[i] * (double)b_orig[i];
    }
    if (!std::isfinite(norm_res) || !std::isfinite(norm_b)) {
        std::cout << "verify: norm_res=" << norm_res << " norm_b=" << norm_b << std::endl;
        return NAN;
    }
    return (float)(std::sqrt(norm_res) / (std::sqrt(norm_b) + 1e-12));
}

int gauss_elimination_scalar(float *A, float *b, int n)
{
    for (int k = 0; k < n; ++k) {
        int maxrow = k;
        float maxval = std::fabs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            float val = std::fabs(A[i * n + k]);
            if (val > maxval) { maxval = val; maxrow = i; }
        }
        if (maxrow != k) {
            for (int j = k; j < n; ++j)
                std::swap(A[k * n + j], A[maxrow * n + j]);
            std::swap(b[k], b[maxrow]);
        }

        float pivot = A[k * n + k];
        if (std::fabs(pivot) < 1e-12f) return -1;
        for (int j = k + 1; j < n; ++j)
            A[k * n + j] /= pivot;
        A[k * n + k] = 1.0f;
        b[k] /= pivot;

        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * n + k];
            for (int j = k + 1; j < n; ++j)
                A[i * n + j] -= A[k * n + j] * factor;
            b[i] -= b[k] * factor;
            A[i * n + k] = 0.0f;
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        float sum = b[i];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i * n + j] * b[j];
        b[i] = sum / A[i * n + i];
    }
    return 0;
}

int gauss_elimination_simd(float *A, float *b, int n)
{
    for (int k = 0; k < n; ++k) {
        int maxrow = k;
        float maxval = std::fabs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            float val = std::fabs(A[i * n + k]);
            if (val > maxval) { maxval = val; maxrow = i; }
        }
        if (maxrow != k) {
            for (int j = k; j < n; ++j)
                std::swap(A[k * n + j], A[maxrow * n + j]);
            std::swap(b[k], b[maxrow]);
        }

        float pivot = A[k * n + k];
        float32x4_t vt = vdupq_n_f32(pivot);

        int j = k + 1;
        for (; j + 3 < n; j += 4) {
            float32x4_t va = vld1q_f32(&A[k * n + j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k * n + j], va);
        }
        for (; j < n; ++j)
            A[k * n + j] /= pivot;
        A[k * n + k] = 1.0f;
        b[k] /= pivot;

        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * n + k];
            float32x4_t vaik = vdupq_n_f32(factor);

            int jj = k + 1;
            for (; jj + 3 < n; jj += 4) {
                float32x4_t vakj = vld1q_f32(&A[k * n + jj]);
                float32x4_t vaij = vld1q_f32(&A[i * n + jj]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i * n + jj], vaij);
            }
            for (; jj < n; ++jj)
                A[i * n + jj] -= A[k * n + jj] * factor;
            b[i] -= b[k] * factor;
            A[i * n + k] = 0.0f;
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        float sum = b[i];
        int j = i + 1;
        for (; j + 3 < n; j += 4) {
            float32x4_t a_vec = vld1q_f32(&A[i * n + j]);
            float32x4_t x_vec = vld1q_f32(&b[j]);
            sum -= vaddvq_f32(vmulq_f32(a_vec, x_vec));
        }
        for (; j < n; ++j)
            sum -= A[i * n + j] * b[j];
        b[i] = sum / A[i * n + i];
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int n = 1024;
    int seed = 114514;
    int num_runs = 5;
    bool scalar = false;

    if (argc >= 2) n = std::stoi(argv[1]);
    if (argc >= 3) seed = std::stoi(argv[2]);
    if (argc >= 4) num_runs = std::stoi(argv[3]);
    if (argc >= 5) scalar = (std::stoi(argv[4]) != 0);

    std::cout << "[build:20260510] Gaussian Elimination"
              << (scalar ? " (scalar baseline)" : " with SIMD (ARM NEON + OpenMP)")
              << std::endl;
    std::cout << "Matrix: " << n << " x " << n << std::endl;
    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;

    std::vector<float> A(n * n), b(n), x_true(n);
    generate_test(A.data(), b.data(), x_true.data(), n, seed);
    std::vector<float> b_orig = b;
    std::vector<float> A_orig = A;

    // 尝试从文件读取
    {
        int file_n;
        std::vector<float> file_A, file_b;
        if (load_from_file("files/input.txt", file_A, file_b, file_n)) {
            A = std::move(file_A); A_orig = A;
            b = std::move(file_b); b_orig = b;
            n = file_n;
            std::cout << "Loaded from files/input.txt, n = " << n << std::endl;
        }
    }

    double total_latency = 0.0;
    bool success = true;

    for (int run = 0; run < num_runs; ++run) {
        std::vector<float> A_run = A_orig;
        std::vector<float> b_run = b_orig;

        auto Start = std::chrono::high_resolution_clock::now();
        int ret = scalar
            ? gauss_elimination_scalar(A_run.data(), b_run.data(), n)
            : gauss_elimination_simd(A_run.data(), b_run.data(), n);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed = End - Start;
        total_latency += elapsed.count();

        if (ret != 0) {
            std::cerr << "Run " << run << ": singular matrix!" << std::endl;
            success = false;
            continue;
        }

        float rel_err = verify_solution(A_orig.data(), b_run.data(), b_orig.data(), n);
        if (!std::isfinite(rel_err) || rel_err > 1e-3f) {
            std::cerr << "Run " << run << ": error too large: " << rel_err << std::endl;
            success = false;
        }
    }

    double avg_latency = total_latency / num_runs;
    std::cout << "Success: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "Average latency: " << avg_latency << " (ms)" << std::endl;

    if (success && num_runs > 0) {
        std::vector<float> A_final = A_orig;
        std::vector<float> b_final = b_orig;
        if (scalar)
            gauss_elimination_scalar(A_final.data(), b_final.data(), n);
        else
            gauss_elimination_simd(A_final.data(), b_final.data(), n);
        save_solution("files/solution.txt", b_final.data(), n);

        float rel_err = verify_solution(A_orig.data(), b_final.data(), b_orig.data(), n);
        if (std::isfinite(rel_err))
            std::cout << "Relative error ||Ax-b||/||b||: " << rel_err << std::endl;
        else
            std::cout << "Relative error ||Ax-b||/||b||: nan (matrix likely singular)" << std::endl;
    }

    return success ? 0 : 1;
}
