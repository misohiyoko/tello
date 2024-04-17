#!/usr/bin/env python3
# coding: utf-8
import cv2
import numpy as np

# ArUcoのライブラリを導入
aruco = cv2.aruco

# 4x4のマーカ, IDは50までの辞書を使用
dictionary = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
pixel = 150
offset = 10
cnt = 9


def generateArMarker():
    # 白いブランク画像を生成
    img = np.zeros((pixel + offset, pixel + offset), dtype=np.uint8)
    img += 255

    x_offset = y_offset = int(offset) // 2
    # 9枚のマーカを作成する
    for i in range(cnt):
        # 150x150ピクセルで画像を作成
        ar_image = aruco.generateImageMarker(dictionary, i, pixel, img, 1)
        # ファイル名の指定
        filename = "ar" + str(i) + ".png"
        # ブランク画像の上にArUcoマーカを重ねる
        # グレースケールからRGBへ変換
        rgb_img = cv2.cvtColor(ar_image, cv2.COLOR_GRAY2RGB)

        # ArUcoマーカの画像を結合
        if (i % 3 == 0):
            hconcat_img = rgb_img
        elif (i % 3 <= 2):
            hconcat_img = cv2.hconcat([hconcat_img, rgb_img])
            if (i % 3 == 2 and i // 3 == 0):
                vconcat_img = hconcat_img
            elif (i % 3 == 2 and i // 3 > 0):
                vconcat_img = cv2.vconcat([vconcat_img, hconcat_img])

        # 1枚ごとのArUcoマーカを出力
        cv2.imwrite(filename, rgb_img)

    # 結合したArUcoマーカを出力
    cv2.imwrite("ar" + str(cnt) + ".png", vconcat_img)


if __name__ == "__main__":
    generateArMarker()