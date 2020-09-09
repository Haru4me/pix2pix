# Image-to-Image при помощи cGAN

Данный репозиторий содержит реализацию Conditional Adversarial Networks для решения задачи "картинка по картинке" на наборе данных facades. Модель реализована на основе U-net генератора и 70x70 PatchGAN детектора(см. орининальную статью).

[[Статья]](https://arxiv.org/pdf/1611.07004.pdf)

<p align="center">
    <b>Пример сгенерированных изображений:</b>
    <br><br>
    <img src="samples/img_1" width="480">
    <img src="samples/img_2" width="480">
    <img src="samples/img_3" width="480">
    <img src="samples/img_4" width="480">
</p>


## Запуск проекта
Для начала скачаем проект:
```
git clone https://github.com/Haru4me/pix2pix.git
```
Для запуска необходимо скачать оригинальный датасэт по [[ссылке]](https://www.kaggle.com/suyashdamle/cyclegan?select=facades) и поместить его в корневую папку проекта. Папка **facades** должна содержать в себе 4 папки (trainA,trainB,testA,testB).

Теперь запустите через командную строку:

```
cd /pix2pix
sh run.sh train
```

После того как модель обучиться, можно будет запустить ее так:

```
cd /pix2pix
sh run.sh test
```

