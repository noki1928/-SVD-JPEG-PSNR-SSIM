# SVD-JPEG-PSNR-SSIM

Этот проект разработан в рамках курса "Императивное программирование" ИИР НГУ. Он реализует два метода сжатия изображений:  
1. Сингулярное разложение (SVD)  
2. JPEG-сжатие (ДКП, квантование, декодирование)  

# Запуск проекта
1. Скачать или скопировать код из репозитория
2. Указать в файле имя изображения в формате bmp, которое хотите сжать
3. В командной строке прописать следующее:

1) cd *Вставить путь к файлу с кодом*
2) gcc svd_jpeg_comparison.c -o svd_jpeg_comparison
3) svd_jpeg.exe (или ./svd_jpeg.exe через запуск PowerShell)

   Для этого должен быть установлен компилятор gcc

# Пример использования
Входное изображение example.bmp. В результате получаются 4 изображения:
1. svd_low.bmp
2. svd_high.bmp
3. dct_low.bmp
4. dct_high.bmp
В терминал выводятся статусы сжатия, а также результат расчета метрик:
SVD compression metrics:
Metrics for svd_high.bmp:
PSNR: 26.90 dB
SSIM: 0.7443
Metrics for svd_low.bmp:
PSNR: 24.00 dB
SSIM: 0.6499
DCT compression metrics:
Metrics for jpg_high.bmp:
PSNR: 26.70 dB
SSIM: 0.8163
Metrics for jpg_low.bmp:
PSNR: 24.14 dB
SSIM: 0.6953


  
   
