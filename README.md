113-1南華大學跨領域-人工智慧 主題:數據增強

數據增強：一種透過應用隨機（但真實）的變換（例如影像旋轉）來增加訓練集多樣性的技術。我們將學習如何透過兩種方式應用數據增強：
使用Keras 預處理層，例如tf.keras.layers.Resizing、tf.keras.layers.Rescaling、tf.keras.layers.RandomFlip和tf.keras.layers.RandomRotation。
使用tf.image方法，例如tf.image.flip_left_right、tf.image.rgb_to_grayscale、tf.image.adjust_brightness、tf.image.central_crop和tf.image.stateless_random*

設定
![image](https://github.com/user-attachments/assets/9fc61a1a-3bbf-406c-8b71-197458985237)

下載數據集
教程使用tf_flowers資料集。為了方便起見，請使用TensorFlow Datasets下載資料集。
![image](https://github.com/user-attachments/assets/7ba24803-d0c7-4477-b5b9-e4c33d686721)

花卉數據集有五個類
![image](https://github.com/user-attachments/assets/674c11e0-331c-4b2c-be0d-40ed3f312567)

我們從數據集中檢索一個影像，然後使用它來演示資料增強。
![image](https://github.com/user-attachments/assets/b0282377-71a3-4e81-a931-0b628b0b8506)

![image](https://github.com/user-attachments/assets/48bc4ce8-5a5f-4729-8910-f85250c9b1f0)

使用Keras 預處理層
調整大小和重新縮放
使用Keras 預處理圖層將影像大小調整為一致的形狀（使用tf.keras.layers.Resizing），並重新調整像素值（使用tf.keras.layers.Rescaling）。
![image](https://github.com/user-attachments/assets/0178673b-ba19-4bac-b6bc-d80f901bcef7)
注意：上面的重新縮放圖層將像素值標準化到[0,1]範圍。如果想要[-1,1]，可以寫tf.keras.layers.Rescaling(1./127.5, offset=-1)。
可以看到將這些圖層應用於影像的結果。
![image](https://github.com/user-attachments/assets/74b45423-2da5-4254-865b-621302b60706)
驗證像素是否在[0, 1]範圍內：
print("Min and max pixel values:", result.numpy().min(), result.numpy().max())

數據增強
可以使用Keras 預處理層進行資料增強，例如tf.keras.layers.RandomFlip和tf.keras.layers.RandomRotation。
我們來建立一些預處理層，然後將它們重複應用於相同影像。









































