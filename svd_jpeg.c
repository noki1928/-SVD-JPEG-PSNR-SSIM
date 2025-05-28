/*
    программа для сжатия изображений методами SVD и JPEG
    реализованы два метода сжатия:
    1. SVD - сингулярное разложение для каждого цветового канала
    2. JPEG - дискретное косинусное преобразование (основные этапы JPEG)
*/
// программа работает без использования сторонних библиотек
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

const char* INPUT_FILE = "example.bmp"; // имя входного файла в формате bmp
const int channels = 3; // количество каналов в изображении
const double PI = 3.14159265358979323846; // число пи

void jpeg_compress(const char* input_file, const char* jpeg_file, const char* output_file, double quality);
void svd_compress(const char* input_file, const char* svd_file, const char* output_file, int k);
void calculate_metrics(const char* original_file, const char* compressed_file);


// структуры для работы с bmp
#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BITMAPFILEHEADER;

typedef struct {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

// ======= функции svd сжатия============
#define MAX_ITER 200 // максимальное количество итераций для вычисления svd
#define TOL 1e-8 // точность вычисления svd

float dot_product(const float *a, const float *b, int len) { // скалярное произведение векторов
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
        sum += a[i] * b[i];
    return sum;
}

float vector_norm(const float *a, int len) { // норма вектора
    return sqrtf(dot_product(a, a, len));
}

void normalize_vector(float *a, int len) { // нормализация вектора
    float norm = vector_norm(a, len);
    if (norm < 1e-8) return;
    for (int i = 0; i < len; i++)
        a[i] /= norm;
}

void mat_vec_mul(const float *A, const float *v, float *result, int m, int n) { // умножение матрицы на вектор
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * v[j];
        result[i] = sum;
    }
}

void matT_vec_mul(const float *A, const float *u, float *result, int m, int n) { // умножение транспонированной матрицы на вектор
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++)
            sum += A[i * n + j] * u[i];
        result[j] = sum;
    }
}

void compute_rank_k_svd(const float *A, int m, int n, int k, float *U, float *S, float *V) { // вычисление svd разложения матрицы A ранга k
    float *A_copy = (float*)malloc(m * n * sizeof(float));
    for (int i = 0; i < m * n; i++)
        A_copy[i] = A[i];

    for (int r = 0; r < k; r++) { // вычисляем k сингулярных троек
        float *v = (float*)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++)
            v[j] = 1.0f;
        normalize_vector(v, n);

        float *u = (float*)malloc(m * sizeof(float));
        float *v_new = (float*)malloc(n * sizeof(float));
        
        // метод степенных итераций для нахождения собственных векторов
        for (int iter = 0; iter < MAX_ITER; iter++) {
            mat_vec_mul(A_copy, v, u, m, n);
            normalize_vector(u, m);
            matT_vec_mul(A_copy, u, v_new, m, n);
            normalize_vector(v_new, n);
            
            // проверяем сходимость
            float diff = 0.0f;
            for (int j = 0; j < n; j++) {
                float d = v_new[j] - v[j];
                diff += d * d;
            }
            diff = sqrtf(diff);
            for (int j = 0; j < n; j++)
                v[j] = v_new[j];
            if (diff < TOL) break;
        }
        
        // вычисляем сингулярное значение
        float *temp = (float*)malloc(m * sizeof(float));
        mat_vec_mul(A_copy, v, temp, m, n);
        float sigma = vector_norm(temp, m);
        free(temp);

        // сохраняем сингулярную тройку
        for (int i = 0; i < m; i++)
            U[i * k + r] = u[i];
        S[r] = sigma;
        for (int j = 0; j < n; j++)
            V[j * k + r] = v[j];

        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A_copy[i * n + j] -= sigma * u[i] * v[j];
            }
        }
        free(v);
        free(v_new);
        free(u);
    }
    free(A_copy);
}

void save_svd_components(const char *filename, int m, int n, int k, // сохранение компонент svd в бинарный файл
                        float *U_R, float *S_R, float *V_R,
                        float *U_G, float *S_G, float *V_G,
                        float *U_B, float *S_B, float *V_B) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Error opening file for writing.\n");
        return;
    }
    fwrite(&m, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&k, sizeof(int), 1, f);
    fwrite(U_R, sizeof(float), m * k, f);
    fwrite(S_R, sizeof(float), k, f);
    fwrite(V_R, sizeof(float), n * k, f);
    fwrite(U_G, sizeof(float), m * k, f);
    fwrite(S_G, sizeof(float), k, f);
    fwrite(V_G, sizeof(float), n * k, f);
    fwrite(U_B, sizeof(float), m * k, f);
    fwrite(S_B, sizeof(float), k, f);
    fwrite(V_B, sizeof(float), n * k, f);
    fclose(f);
    printf("SVD components saved to %s\n", filename);
}

void load_svd_components(const char *filename, int *m, int *n, int *k, // загрузка компонент svd из бинарного файла
                         float **U_R, float **S_R, float **V_R,
                         float **U_G, float **S_G, float **V_G,
                         float **U_B, float **S_B, float **V_B) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("Error opening file for reading.\n");
        return;
    }
    fread(m, sizeof(int), 1, f);
    fread(n, sizeof(int), 1, f);
    fread(k, sizeof(int), 1, f);
    *U_R = (float*)malloc((*m) * (*k) * sizeof(float));
    *S_R = (float*)malloc((*k) * sizeof(float));
    *V_R = (float*)malloc((*n) * (*k) * sizeof(float));
    *U_G = (float*)malloc((*m) * (*k) * sizeof(float));
    *S_G = (float*)malloc((*k) * sizeof(float));
    *V_G = (float*)malloc((*n) * (*k) * sizeof(float));
    *U_B = (float*)malloc((*m) * (*k) * sizeof(float));
    *S_B = (float*)malloc((*k) * sizeof(float));
    *V_B = (float*)malloc((*n) * (*k) * sizeof(float));
    fread(*U_R, sizeof(float), (*m) * (*k), f);
    fread(*S_R, sizeof(float), *k, f);
    fread(*V_R, sizeof(float), (*n) * (*k), f);
    fread(*U_G, sizeof(float), (*m) * (*k), f);
    fread(*S_G, sizeof(float), *k, f);
    fread(*V_G, sizeof(float), (*n) * (*k), f);
    fread(*U_B, sizeof(float), (*m) * (*k), f);
    fread(*S_B, sizeof(float), *k, f);
    fread(*V_B, sizeof(float), (*n) * (*k), f);
    fclose(f);
    printf("SVD components loaded from %s\n", filename);
}

// ========== функции для нескольких этапов сжатия jpeg=========
// сразу хочу отметить, что здесь представлено не полное сжатие jpeg, а только несколько этапов, по-скольку это не цель данного проекта
// здесь реализованы функции для обработки цветовых каналов, косинусного преобразования, квантования, а также их обратные

// таблицы квантования взяты из стандартной реализации jpeg
const int QUANT_TABLE_Y[64] = { // таблица квантования для яркости
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};

const int QUANT_TABLE_C[64] = { // таблица квантования для цветности
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99
};

void rgb_to_ycbcr(int width, int height, uint8_t *rgb, uint8_t *ycbcr) { // преобразование из rgb в ycbcr
    for (int i = 0; i < height * width; i++) {

        int R = rgb[3 * i];
        int G = rgb[3 * i + 1];
        int B = rgb[3 * i + 2];
        
        // формулы преобразования из стандарта jpeg
        int Y  = (int)(0.299 * R + 0.587 * G + 0.114 * B);
        int Cb = (int)(-0.168736 * R - 0.331264 * G + 0.5 * B + 128);
        int Cr = (int)(0.5 * R - 0.418688 * G - 0.081312 * B + 128);
        
        // ограничиваем значения в диапазоне 0-255
        ycbcr[3 * i]     = (uint8_t)fmin(fmax(Y, 0), 255);
        ycbcr[3 * i + 1] = (uint8_t)fmin(fmax(Cb, 0), 255);
        ycbcr[3 * i + 2] = (uint8_t)fmin(fmax(Cr, 0), 255);
    }
}

void ycbcr_to_rgb(int width, int height, uint8_t *ycbcr, uint8_t *rgb) { // преобразование из ycbcr в rgb, аналогична предыдущей функции
    for (int i = 0; i < height * width; i++) {

        double Y = ycbcr[3 * i];
        double Cb = ycbcr[3 * i + 1] - 128.0;
        double Cr = ycbcr[3 * i + 2] - 128.0;

        int R = (int)(Y + 1.402 * Cr);
        int G = (int)(Y - 0.344136 * Cb - 0.714136 * Cr);
        int B = (int)(Y + 1.772 * Cb);
        
        rgb[3 * i]     = (uint8_t)fmin(fmax(R, 0), 255);
        rgb[3 * i + 1] = (uint8_t)fmin(fmax(G, 0), 255);
        rgb[3 * i + 2] = (uint8_t)fmin(fmax(B, 0), 255);
    }
}

void dct(double *block) { //дискретное косинусное преобразование
    double temp[64] = {0};
    
    for(int u = 0; u < 8; u++) {
        for(int v = 0; v < 8; v++) {
            double sum = 0.0;
            double cu = 1.0;
            double cv = 1.0;
            if(u == 0) {
                cu = 1.0 / sqrt(2.0);
            }
            if(v == 0) {
                cv = 1.0 / sqrt(2.0);
            }
            
            // вычисление dct по формуле
            for(int x = 0; x < 8; x++) {
                for(int y = 0; y < 8; y++) {
                    double pixel = block[x * 8 + y];
                    double cos_x = cos((2.0 * x + 1.0) * u * PI / 16.0);
                    double cos_y = cos((2.0 * y + 1.0) * v * PI / 16.0);
                    sum += pixel * cos_x * cos_y;
                }
            }
            
            temp[u * 8 + v] = 0.25 * cu * cv * sum;
        }
    }
    
    memcpy(block, temp, 64 * sizeof(double));
}

void idct(double *block) { // обратное дискретное косинусное преобразование
    double temp[64] = {0};
    
    for(int x = 0; x < 8; x++) {
        for(int y = 0; y < 8; y++) {
            double sum = 0.0;
            
            for(int u = 0; u < 8; u++) {
                for(int v = 0; v < 8; v++) {
                    double cu = 1.0;
                    double cv = 1.0;
                    if(u == 0) {
                        cu = 1.0 / sqrt(2.0);
                    }
                    if(v == 0) {
                        cv = 1.0 / sqrt(2.0);
                    }
                    
                    double coeff = block[u * 8 + v];
                    double cos_x = cos((2.0 * x + 1.0) * u * PI / 16.0);
                    double cos_y = cos((2.0 * y + 1.0) * v * PI / 16.0);
                    sum += cu * cv * coeff * cos_x * cos_y;
                }
            }
            
            temp[x * 8 + y] = 0.25 * sum;
        }
    }
    
    memcpy(block, temp, 64 * sizeof(double));
}

void quantize(double *block, double quality, int is_chroma) { // квантование
    double scale = fmax(0.01, fmin(1.0, quality));
    
    
    const int* quant_table; // выбор таблицы квантования
    if (is_chroma) {
        quant_table = QUANT_TABLE_C;
    } else {
        quant_table = QUANT_TABLE_Y;
    }
    
    
    for(int i = 0; i < 64; i++) { // делим каждый коэффициент на соответствующее значение из таблицы
        double q = quant_table[i] / scale;
        block[i] = round(block[i] / q);
    }
}

void dequantize(double *block, double quality, int is_chroma) { // обратное квантование
    
    double scale = fmax(0.01, fmin(1.0, quality));
    
    // выбираем таблицу квантования
    const int* quant_table;
    if (is_chroma) {
        quant_table = QUANT_TABLE_C;
    } else {
        quant_table = QUANT_TABLE_Y;
    }
    
    // умножаем каждый коэффициент на соответствующее значение из таблицы
    for(int i = 0; i < 64; i++) {
        double q = quant_table[i] / scale;
        block[i] = block[i] * q;
    }
}


// ==================метрики качества ===============
// функции для вычисления метрик качества сжатого изображения
// реализованы две метрики:
// PSNR (Peak Signal-to-Noise Ratio)
// SSIM (Structural Similarity Index)

double mean(const uint8_t* x, int size) { // вычисление среднего значения массива
    double sum = 0.0;
    for(int i = 0; i < size; i++) {
        sum += x[i];
    }
    return sum / size;
}

double variance(const uint8_t* x, double mean_x, int size) { // вычисление дисперсии массива
    double sum = 0.0;
    for(int i = 0; i < size; i++) {
        double diff = x[i] - mean_x;
        sum += diff * diff;
    }
    return sum / (size - 1);
}

double covariance(const uint8_t* x, const uint8_t* y, double mean_x, double mean_y, int size) { // вычисление ковариации двух массивов
    double sum = 0.0;
    for(int i = 0; i < size; i++) {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return sum / (size - 1);
}

double calculate_psnr(const uint8_t* original, const uint8_t* compressed, int width, int height, int channels) { // вычисление PSNR
    double mse = 0.0;
    int size = width * height * channels;
    
    // MSE
    for(int i = 0; i < size; i++) {
        double diff = (double)original[i] - (double)compressed[i];
        mse += diff * diff;
    }
    
    mse /= size;
    
    if(mse == 0) return -1; 
    
    // вычисляем psnr по формуле
    double max_value = 255.0;
    return 20 * log10(max_value) - 10 * log10(mse);
}

double calculate_ssim(const uint8_t* original, const uint8_t* compressed, int width, int height, int channels) { // вычисление SSIM
    const double K1 = 0.01;
    const double K2 = 0.03;
    const double L = 255.0;
    const double C1 = (K1 * L) * (K1 * L);
    const double C2 = (K2 * L) * (K2 * L);
    
    double ssim_sum = 0.0;
    int window_size = 8;
    int num_windows = (height / window_size) * (width / window_size) * channels;
    
    
    for(int c = 0; c < channels; c++) { // проходим по всем окнам в изображении
        for(int i = 0; i <= height - window_size; i += window_size) {
            for(int j = 0; j <= width - window_size; j += window_size) {
                uint8_t* window1 = (uint8_t*)malloc(window_size * window_size);
                uint8_t* window2 = (uint8_t*)malloc(window_size * window_size);
                
                // копируем данные окна
                int idx = 0;
                for(int y = 0; y < window_size; y++) {
                    for(int x = 0; x < window_size; x++) {
                        int pos = ((i + y) * width + (j + x)) * channels + c;
                        window1[idx] = original[pos];
                        window2[idx] = compressed[pos];
                        idx++;
                    }
                }
                
                // вычисляем локальные характеристики
                double mean1 = mean(window1, window_size * window_size);
                double mean2 = mean(window2, window_size * window_size);
                double var1 = variance(window1, mean1, window_size * window_size);
                double var2 = variance(window2, mean2, window_size * window_size);
                double covar = covariance(window1, window2, mean1, mean2, window_size * window_size);
                
                // вычисляем SSIM для текущего окна по формуле
                double numerator = (2 * mean1 * mean2 + C1) * (2 * covar + C2);
                double denominator = (mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2);
                double ssim_window = numerator / denominator;
                
                ssim_sum += ssim_window;
                
                free(window1);
                free(window2);
            }
        }
    }
    
    // возвращаем среднее значение SSIM по всем окнам
    return ssim_sum / num_windows;
}


// ================== main ==============
int main() {
    // сжатие svd
    printf("\nPerforming SVD compression...\n");
    svd_compress(INPUT_FILE, "svd_high.bin", "svd_high.bmp", 100); // 100
    svd_compress(INPUT_FILE, "svd_low.bin", "svd_low.bmp", 50);    // 50
    
    // сжатие jpeg
    printf("\nPerforming JPEG compression...\n");
    jpeg_compress(INPUT_FILE, "jpeg_high.bin", "jpg_high.bmp", 0.9); // 90%
    jpeg_compress(INPUT_FILE, "jpeg_low.bin", "jpg_low.bmp", 0.3);   // 30%
    
    // метрики качества
    printf("\nCalculating quality metrics...\n");
    printf("\nSVD compression metrics:\n");
    calculate_metrics(INPUT_FILE, "svd_high.bmp"); // метрики SVD высокого качества
    calculate_metrics(INPUT_FILE, "svd_low.bmp");  // метрики SVD низкого качества
    
    printf("\nJPEG compression metrics:\n");
    calculate_metrics(INPUT_FILE, "jpg_high.bmp"); // метрики JPEG высокого качества
    calculate_metrics(INPUT_FILE, "jpg_low.bmp");  // метрики JPEG низкого качества
    
    return 0;
}

void jpeg_compress(const char* input_file, const char* jpeg_file, const char* output_file, double quality) { // сжатие изображения JPEG
    // открываем входной файл
    FILE* file = fopen(input_file, "rb");
    if (!file) {
        printf("Error opening BMP file.\n");
        return;
    }

    BITMAPFILEHEADER file_header;
    BITMAPINFOHEADER info_header;

    fread(&file_header, sizeof(BITMAPFILEHEADER), 1, file);
    if (file_header.bfType != 0x4D42) { // проверяем сигнатуру BMP
        printf("Not a valid BMP file.\n");
        fclose(file);
        return;
    }

    fread(&info_header, sizeof(BITMAPINFOHEADER), 1, file);
    if (info_header.biBitCount != 24 || info_header.biCompression != 0) { // проверяем формат
        printf("Only 24-bit uncompressed BMP files are supported.\n");
        fclose(file);
        return;
    }

    int width = info_header.biWidth;
    int height = abs(info_header.biHeight);
    int row_padded = (width * 3 + 3) & (~3);
    int padding = row_padded - (width * 3);

    uint8_t* rgb = (uint8_t*)malloc(width * height * 3);
    if (!rgb) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return;
    }

    // читаем данные изображения
    fseek(file, file_header.bfOffBits, SEEK_SET);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            fread(&rgb[idx + 2], 1, 1, file);
            fread(&rgb[idx + 1], 1, 1, file);
            fread(&rgb[idx], 1, 1, file);
        }
        fseek(file, padding, SEEK_CUR);
    }
    fclose(file);

    // преобразуем RGB в YCbCr
    uint8_t *ycbcr = (uint8_t *)malloc(width * height * channels);
    rgb_to_ycbcr(width, height, rgb, ycbcr);
    
    // выделяем память для каналов YCbCr
    double *y = (double *)malloc(width * height * sizeof(double));
    double *cb = (double *)malloc(width * height * sizeof(double));
    double *cr = (double *)malloc(width * height * sizeof(double));

    // центрируем данные
    for(int i = 0; i < width * height; i++) {
        y[i] = (double)ycbcr[3 * i] - 128.0;
        cb[i] = (double)ycbcr[3 * i + 1] - 128.0;
        cr[i] = (double)ycbcr[3 * i + 2] - 128.0;
    }
    
    // открываем файл для сжатых данных
    FILE *compressed = fopen(jpeg_file, "wb");
    if (!compressed) {
        printf("Error opening compressed file!\n");
        return;
    }

    // записываем размеры и параметр качества
    fwrite(&width, sizeof(int), 1, compressed);
    fwrite(&height, sizeof(int), 1, compressed);
    fwrite(&quality, sizeof(double), 1, compressed);

    double block[64];
    
    // обрабатываем каждый канал отдельно
    for(int channel = 0; channel < 3; channel++) {
        double *current_channel;
        int is_chroma;
        
        // выбираем текущий канал
        if (channel == 0) {
            current_channel = y;
            is_chroma = 0;
        } else if (channel == 1) {
            current_channel = cb;
            is_chroma = 1;
        } else {
            current_channel = cr;
            is_chroma = 1;
        }
        
        int block_rows = (height + 7) / 8;
        int block_cols = (width + 7) / 8;
        
        for(int i = 0; i < block_rows; i++) {
            for(int j = 0; j < block_cols; j++) {
                for(int m = 0; m < 8; m++) {
                    for(int n = 0; n < 8; n++) {
                        int row = i * 8 + m;
                        int col = j * 8 + n;
                        
                        if(row < height && col < width) {
                            block[m * 8 + n] = current_channel[row * width + col];
                        } else {
                            // зеркалируем края для блоков на границе
                            int mirror_row = row >= height ? 2 * height - row - 1 : row;
                            int mirror_col = col >= width ? 2 * width - col - 1 : col;
                            block[m * 8 + n] = current_channel[mirror_row * width + mirror_col];
                        }
                    }
                }
                
                dct(block);
                quantize(block, quality, is_chroma);
                fwrite(block, sizeof(double), 64, compressed);
            }
        }
    }
    
    fclose(compressed);
    
    // декомпрессия
    compressed = fopen(jpeg_file, "rb");
    if (!compressed) {
        printf("Error opening compressed file!\n");
        return;
    }
    
    int width_read, height_read;
    double quality_read;
    fread(&width_read, sizeof(int), 1, compressed);
    fread(&height_read, sizeof(int), 1, compressed);
    fread(&quality_read, sizeof(double), 1, compressed);
    
    double *y_dec = (double *)malloc(width * height * sizeof(double));
    double *cb_dec = (double *)malloc(width * height * sizeof(double));
    double *cr_dec = (double *)malloc(width * height * sizeof(double));
    
    uint8_t *ycbcr_dec = (uint8_t *)malloc(width * height * channels);
    uint8_t *rgb_dec = (uint8_t *)malloc(width * height * channels);
    
    for(int channel = 0; channel < 3; channel++) {
        double *current_channel;
        int is_chroma;
        if (channel == 0) {
            current_channel = y_dec;
            is_chroma = 0;
        } else if (channel == 1) {
            current_channel = cb_dec;
            is_chroma = 1;
        } else {
            current_channel = cr_dec;
            is_chroma = 1;
        }
        
        int block_rows = (height + 7) / 8;
        int block_cols = (width + 7) / 8;
        
        for(int i = 0; i < block_rows; i++) {
            for(int j = 0; j < block_cols; j++) {
                fread(block, sizeof(double), 64, compressed);
                
                dequantize(block, quality, is_chroma);
                idct(block);
                
                for(int m = 0; m < 8; m++) {
                    for(int n = 0; n < 8; n++) {
                        int row = i * 8 + m;
                        int col = j * 8 + n;
                        if(row < height && col < width) {
                            current_channel[row * width + col] = block[m * 8 + n];
                        }
                    }
                }
            }
        }
    }
    
    for(int i = 0; i < width * height; i++) {
        ycbcr_dec[3 * i] = (uint8_t)fmin(fmax(y_dec[i] + 128.0, 0), 255);
        ycbcr_dec[3 * i + 1] = (uint8_t)fmin(fmax(cb_dec[i] + 128.0, 0), 255);
        ycbcr_dec[3 * i + 2] = (uint8_t)fmin(fmax(cr_dec[i] + 128.0, 0), 255);
    }
    
    // преобразуем обратно в RGB
    ycbcr_to_rgb(width, height, ycbcr_dec, rgb_dec);
    
    // создаем выходной BMP файл
    FILE *outFile = fopen(output_file, "wb");
    if (!outFile) {
        printf("Error creating output file!\n");
        return;
    }
    

    BITMAPFILEHEADER out_file_header = {0};
    BITMAPINFOHEADER out_info_header = {0};
    
    out_file_header.bfType = 0x4D42;
    out_file_header.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (width * 3 + padding) * height;
    out_file_header.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    
    out_info_header.biSize = sizeof(BITMAPINFOHEADER);
    out_info_header.biWidth = width;
    out_info_header.biHeight = height;
    out_info_header.biPlanes = 1;
    out_info_header.biBitCount = 24;
    out_info_header.biCompression = 0;
    out_info_header.biSizeImage = (width * 3 + padding) * height;
    out_info_header.biXPelsPerMeter = 2835;
    out_info_header.biYPelsPerMeter = 2835;
    
    fwrite(&out_file_header, sizeof(BITMAPFILEHEADER), 1, outFile);
    fwrite(&out_info_header, sizeof(BITMAPINFOHEADER), 1, outFile);
    
    // записываем данные пикселей
    unsigned char pad[3] = {0, 0, 0};
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            unsigned char pixel[3];
            pixel[0] = rgb_dec[idx * 3 + 2];
            pixel[1] = rgb_dec[idx * 3 + 1];
            pixel[2] = rgb_dec[idx * 3];
            fwrite(pixel, 3, 1, outFile);
        }
        if (padding > 0) {
            fwrite(pad, 1, padding, outFile);
        }
    }
    
    fclose(outFile);
    free(y);
    free(cb);
    free(cr);
    free(ycbcr);
    free(rgb);
    free(y_dec);
    free(cb_dec);
    free(cr_dec);
    free(ycbcr_dec);
    free(rgb_dec);
}

void svd_compress(const char* input_file, const char* svd_file, const char* output_file, int k) { // сжатие изображения svd

    FILE* file = fopen(input_file, "rb");
    if (!file) {
        printf("Error opening BMP file.\n");
        return;
    }

    // читаем заголовки BMP файла
    BITMAPFILEHEADER file_header;
    BITMAPINFOHEADER info_header;

    fread(&file_header, sizeof(BITMAPFILEHEADER), 1, file);
    if (file_header.bfType != 0x4D42) { // проверяем сигнатуру BMP
        printf("Not a valid BMP file.\n");
        fclose(file);
        return;
    }

    fread(&info_header, sizeof(BITMAPINFOHEADER), 1, file);
    if (info_header.biBitCount != 24 || info_header.biCompression != 0) { // проверяем формат
        printf("Only 24-bit uncompressed BMP files are supported.\n");
        fclose(file);
        return;
    }

    // получаем размеры изображения
    int width = info_header.biWidth;
    int height = abs(info_header.biHeight);
    int m = height, n = width;

    // вычисляем размер строки с учетом выравнивания
    int row_padded = (width * 3 + 3) & (~3);
    int padding = row_padded - (width * 3);

    // выделяем память для данных изображения
    unsigned char* img_data = (unsigned char*)malloc(width * height * 3);
    if (!img_data) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return;
    }

    // читаем данные изображения
    fseek(file, file_header.bfOffBits, SEEK_SET);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            fread(&img_data[idx + 2], 1, 1, file);
            fread(&img_data[idx + 1], 1, 1, file);
            fread(&img_data[idx], 1, 1, file);
        }
        fseek(file, padding, SEEK_CUR); // пропускаем padding
    }
    fclose(file);


    float *A_R = (float*)malloc(m * n * sizeof(float));
    float *A_G = (float*)malloc(m * n * sizeof(float));
    float *A_B = (float*)malloc(m * n * sizeof(float));
    
    // нормализуем входные данные в диапазон 0-1
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            int img_idx = idx * 3;
            A_R[idx] = (float)img_data[img_idx] / 255.0f;
            A_G[idx] = (float)img_data[img_idx + 1] / 255.0f;
            A_B[idx] = (float)img_data[img_idx + 2] / 255.0f;
        }
    }
    
    // выделяем память для компонент SVD
    float *U_R = (float*)malloc(m * k * sizeof(float));
    float *S_R = (float*)malloc(k * sizeof(float));
    float *V_R = (float*)malloc(n * k * sizeof(float));
    float *U_G = (float*)malloc(m * k * sizeof(float));
    float *S_G = (float*)malloc(k * sizeof(float));
    float *V_G = (float*)malloc(n * k * sizeof(float));
    float *U_B = (float*)malloc(m * k * sizeof(float));
    float *S_B = (float*)malloc(k * sizeof(float));
    float *V_B = (float*)malloc(n * k * sizeof(float));
    
    // вычисляем SVD для каждого канала
    compute_rank_k_svd(A_R, m, n, k, U_R, S_R, V_R);
    compute_rank_k_svd(A_G, m, n, k, U_G, S_G, V_G);
    compute_rank_k_svd(A_B, m, n, k, U_B, S_B, V_B);

    // сохраняем компоненты SVD
    save_svd_components(svd_file, m, n, k, U_R, S_R, V_R, U_G, S_G, V_G, U_B, S_B, V_B);
    
    // освобождаем память компонент SVD
    free(U_R); free(S_R); free(V_R);
    free(U_G); free(S_G); free(V_G);
    free(U_B); free(S_B); free(V_B);
    
    // загружаем компоненты SVD для восстановления
    float *U_R_load, *S_R_load, *V_R_load;
    float *U_G_load, *S_G_load, *V_G_load;
    float *U_B_load, *S_B_load, *V_B_load;
    int m_load, n_load, k_load;
    load_svd_components(svd_file, &m_load, &n_load, &k_load,
                        &U_R_load, &S_R_load, &V_R_load,
                        &U_G_load, &S_G_load, &V_G_load,
                        &U_B_load, &S_B_load, &V_B_load);
    
    // проверяем размеры
    if (m_load != m || n_load != n || k_load != k) {
        printf("loaded data dimensions don't match.\n");
        return;
    }
    
    // выделяем память для восстановленных матриц
    float *A_R_approx = (float*)calloc(m * n, sizeof(float));
    float *A_G_approx = (float*)calloc(m * n, sizeof(float));
    float *A_B_approx = (float*)calloc(m * n, sizeof(float));
    
    // восстанавливаем изображение
    for (int r = 0; r < k; r++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                A_R_approx[idx] += S_R_load[r] * U_R_load[i * k + r] * V_R_load[j * k + r];
                A_G_approx[idx] += S_G_load[r] * U_G_load[i * k + r] * V_G_load[j * k + r];
                A_B_approx[idx] += S_B_load[r] * U_B_load[i * k + r] * V_B_load[j * k + r];
            }
        }
    }
    
    // преобразуем обратно в диапазон 0-255
    unsigned char *out_img = (unsigned char*)malloc(m * n * 3 * sizeof(unsigned char));
    for (int i = 0; i < m * n; i++) {
        int r = (int)roundf(A_R_approx[i] * 255.0f);
        int g = (int)roundf(A_G_approx[i] * 255.0f);
        int b = (int)roundf(A_B_approx[i] * 255.0f);
        if (r < 0) r = 0; if (r > 255) r = 255;
        if (g < 0) g = 0; if (g > 255) g = 255;
        if (b < 0) b = 0; if (b > 255) b = 255;
        out_img[i * 3]     = (unsigned char)r;
        out_img[i * 3 + 1] = (unsigned char)g;
        out_img[i * 3 + 2] = (unsigned char)b;
    }
    
    // создаем выходной BMP файл
    FILE *outFile = fopen(output_file, "wb");
    if (!outFile) {
        printf("Error creating output file!\n");
        return;
    }
    

    BITMAPFILEHEADER out_file_header = {0};
    BITMAPINFOHEADER out_info_header = {0};
    
    out_file_header.bfType = 0x4D42; // сигнатура BMP
    out_file_header.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (width * 3 + padding) * height;
    out_file_header.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    
    out_info_header.biSize = sizeof(BITMAPINFOHEADER);
    out_info_header.biWidth = width;
    out_info_header.biHeight = height;
    out_info_header.biPlanes = 1;
    out_info_header.biBitCount = 24;
    out_info_header.biCompression = 0;
    out_info_header.biSizeImage = (width * 3 + padding) * height;
    out_info_header.biXPelsPerMeter = 2835;
    out_info_header.biYPelsPerMeter = 2835;
    

    fwrite(&out_file_header, sizeof(BITMAPFILEHEADER), 1, outFile);
    fwrite(&out_info_header, sizeof(BITMAPINFOHEADER), 1, outFile);
    
    // записываем данные пикселей
    unsigned char pad[3] = {0, 0, 0};
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            unsigned char pixel[3];
            pixel[0] = out_img[idx * 3 + 2];
            pixel[1] = out_img[idx * 3 + 1];
            pixel[2] = out_img[idx * 3];
            fwrite(pixel, 3, 1, outFile);
        }
        if (padding > 0) {
            fwrite(pad, 1, padding, outFile);
        }
    }
    
 
    fclose(outFile);
    free(A_R);
    free(A_G);
    free(A_B);
    free(U_R_load);
    free(S_R_load);
    free(V_R_load);
    free(U_G_load);
    free(S_G_load);
    free(V_G_load);
    free(U_B_load);
    free(S_B_load);
    free(V_B_load);
    free(A_R_approx);
    free(A_G_approx);
    free(A_B_approx);
    free(out_img);
    free(img_data);
}


void calculate_metrics(const char* original_file, const char* compressed_file) {

    BITMAPFILEHEADER orig_header;
    BITMAPINFOHEADER orig_info;
    uint8_t* original = NULL;
    uint8_t* compressed = NULL;
    uint8_t* row = NULL;
    FILE* f = NULL;
    
    f = fopen(original_file, "rb");
    if(!f) {
        printf("Error opening original file: %s\n", original_file);
        return;
    }
    
    if(fread(&orig_header, sizeof(BITMAPFILEHEADER), 1, f) != 1 ||
       fread(&orig_info, sizeof(BITMAPINFOHEADER), 1, f) != 1) {
        printf("Error reading headers from original file\n");
        fclose(f);
        return;
    }
    
    int width = orig_info.biWidth;
    int height = abs(orig_info.biHeight);
    int row_padded = (width * 3 + 3) & (~3);
    
    original = (uint8_t*)malloc(width * height * 3);
    if(!original) {
        printf("Memory allocation failed!\n");
        fclose(f);
        return;
    }
    
    fseek(f, orig_header.bfOffBits, SEEK_SET);
    row = (uint8_t*)malloc(row_padded);
    if(!row) {
        printf("Memory allocation failed!\n");
        free(original);
        fclose(f);
        return;
    }
    
    for(int y = height - 1; y >= 0; y--) {
        if(fread(row, 1, row_padded, f) != row_padded) {
            printf("Error reading original image data\n");
            free(row);
            free(original);
            fclose(f);
            return;
        }
        for(int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            original[idx + 2] = row[x * 3];
            original[idx + 1] = row[x * 3 + 1];
            original[idx] = row[x * 3 + 2];
        }
    }
    
    free(row);
    fclose(f);
    
    BITMAPFILEHEADER comp_header;
    BITMAPINFOHEADER comp_info;
    
    f = fopen(compressed_file, "rb");
    if(!f) {
        printf("Error opening compressed file: %s\n", compressed_file);
        free(original);
        return;
    }
    
    if(fread(&comp_header, sizeof(BITMAPFILEHEADER), 1, f) != 1 ||
       fread(&comp_info, sizeof(BITMAPINFOHEADER), 1, f) != 1) {
        printf("Error reading headers from compressed file\n");
        free(original);
        fclose(f);
        return;
    }
    
    if(comp_info.biWidth != width || abs(comp_info.biHeight) != height) {
        printf("Error: compressed image dimensions don't match original!\n");
        free(original);
        fclose(f);
        return;
    }
    
    compressed = (uint8_t*)malloc(width * height * 3);
    if(!compressed) {
        printf("Memory allocation failed!\n");
        free(original);
        fclose(f);
        return;
    }
    
    fseek(f, comp_header.bfOffBits, SEEK_SET);
    row = (uint8_t*)malloc(row_padded);
    if(!row) {
        printf("Memory allocation failed!\n");
        free(compressed);
        free(original);
        fclose(f);
        return;
    }
    
    for(int y = height - 1; y >= 0; y--) {
        if(fread(row, 1, row_padded, f) != row_padded) {
            printf("Error reading compressed image data\n");
            free(row);
            free(compressed);
            free(original);
            fclose(f);
            return;
        }
        for(int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            compressed[idx + 2] = row[x * 3];
            compressed[idx + 1] = row[x * 3 + 1];
            compressed[idx] = row[x * 3 + 2];
        }
    }
    
    free(row);
    fclose(f);
    

    double psnr = calculate_psnr(original, compressed, width, height, 3);
    double ssim = calculate_ssim(original, compressed, width, height, 3);
    
    printf("\nMetrics for %s:\n", compressed_file);
    printf("PSNR: %.2f dB\n", psnr);
    printf("SSIM: %.4f\n", ssim);
    
    free(original);
    free(compressed);
}
