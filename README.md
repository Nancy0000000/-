113-1南華大學跨領域-人工智慧 主題:數據增強
報告學生:11220016陳靖尹 11218102蘇韋綾

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
![image](https://github.com/user-attachments/assets/c2085cc6-4145-4a8b-a833-70393d2b669a)

數據增強
可以使用Keras 預處理層進行數據增強，例如tf.keras.layers.RandomFlip和tf.keras.layers.RandomRotation。
我們來建立一些預處理層，然後將它們重複應用於相同影像。
![image](https://github.com/user-attachments/assets/e41e82ee-7e4b-4c9d-a75b-a6cefc6dd444)
有多種預處理層可用於數據增強，包括tf.keras.layers.RandomContrast、tf.keras.layers.RandomCrop、tf.keras.layers.RandomZoom等。

使用Keras 預處理層的兩個選項
您可以透過兩種方式使用這些預處理層，但需進行重要的權衡。
選項1：使預處理層成為模型的一部分
![image](https://github.com/user-attachments/assets/3282dbd0-e5c9-4e6b-9832-ef5dfeefbfc6)
在這種情況下，需要注意兩個要點：
資料增強將與其他層在裝置端同步運行，並受益於GPU 加速。
使用model.save匯出模型時，預處理層將與模型的其他部分一起儲存。如果稍後部署此模型，它將自動標準化映像（根據您的層配置）。這可以省去在伺服器端重新實作該邏輯的工作。
注意：資料增強在測試時處於停用狀態，因此只有在呼叫Model.fit（而非Model.evaluate或Model.predict）期間才會對輸入影像進行增強。

選項2：將預處理層應用於資料集
![image](https://github.com/user-attachments/assets/f1381cd0-aa81-40f2-8370-dcdce65761d5)
透過這種方式，您可以使用Dataset.map建立產生增強影像批次的資料集。在本例中：
資料增強將在CPU 上非同步進行，且為非阻塞性。您可以使用Dataset.prefetch將GPU 上的模型訓練與資料資料預處理重疊，如下所示。
在本例中，當呼叫Model.save時，預處理層將不會隨模型一起匯出。在保存模型或在伺服器端重新實作它們之前，您需要將它們附加到模型上。訓練後，您可以在匯出之前附加預處理層。

將預處理層應用於資料集
使用上面建立的Keras 預處理層配置訓練數據集、數據資料集和測試數據集。您還將配置數據集以提高效能，具體方式是使用並行讀取和緩衝預提取從磁碟產生批次，這樣不會阻塞I/O。 （您可以透過使用tf.data API 提高效能指南來詳細了解資料集效能）。
註：應僅對訓練集應用數據增強。
![image](https://github.com/user-attachments/assets/b946e973-b50b-40db-8a78-f7dfa5573d31)

訓練模型
為了完整起見，您現在將使用剛剛準備的數據集訓練模型。
序貫模型由三個卷積塊( tf.keras.layers.Conv2D) 組成，每個卷積塊都有一個最大池化層( tf.keras.layers.MaxPooling2D)。有一個全連接層( tf.keras.layers.Dense)，上面有128 個單元，由ReLU 激活函數( 'relu') 激活。此模型尚未針對準確率進行調整（目標是展示機制）。
![image](https://github.com/user-attachments/assets/0ec87dfa-43f2-4652-a25b-dafbcb9d3c1d)

選擇tf.keras.optimizers.Adam最佳化器和tf.keras.losses.SparseCategoricalCrossentropy損失函數。若要查看每個訓練週期的訓練和驗證準確率，請將metrics參數傳遞給Model.compile。
![image](https://github.com/user-attachments/assets/a4ce74d8-854f-41e1-8058-0c9018e3e387)
訓練幾個週期：
![image](https://github.com/user-attachments/assets/c041ef4e-96cf-4916-9692-8d96aaafd6b6)

自訂數據增強
也可以建立自訂數據增強層。
教程的這一部分展示了兩種操作方式：
首先，建立一個tf.keras.layers.Lambda層。這是編寫簡潔程式碼的好方式。
接下來，將透過子類化編寫一個新層，這會給您更多的控制。
兩個層都會根據某種機率隨機反轉影像中的顏色。
![image](https://github.com/user-attachments/assets/d6f1e52e-92d1-424a-b497-3136e8113f6d)
![image](https://github.com/user-attachments/assets/2d55d9a6-7712-4e33-ae8f-90c0f689ea64)

接下來，透過子類化實現自訂層：
![image](https://github.com/user-attachments/assets/0a89c59d-cfe0-4255-95a1-216c8f36664a)

使用tf.image
上述Keras 預訓練實用工具十分方便。但為了更精細的控制，您可以使用tf.data和tf.image編寫自己的資料增強管線或資料增強層。您也可以查看TensorFlow Addons 影像：運算和TensorFlow I/O：色彩空間轉換。
由於花卉資料集之前已經配置了資料增強，因此我們將其重新導入以重新開始。
![image](https://github.com/user-attachments/assets/60c95dbc-6c2a-4c53-9f60-d89ab4377bfe)
檢索一個影像以供使用：
![image](https://github.com/user-attachments/assets/b09eb497-5bce-4717-bac2-f2b46866fb84)
![image](https://github.com/user-attachments/assets/a52243a5-1696-4684-a6bf-4df12392ee23)
我們來使用以下函數呈現原始影像和增強影像，然後並排比較。
![image](https://github.com/user-attachments/assets/138e868f-03ed-4705-80cf-34b4f77fe1f1)

數據增強
翻轉影像
使用tf.image.flip_left_right垂直或水平翻轉影像：
![image](https://github.com/user-attachments/assets/8a2b9849-0adf-4e30-a071-f49488ef5f13)
![image](https://github.com/user-attachments/assets/167a374f-9160-4a80-8c6b-5168297d80d1)
對影像進行灰階處理
您可以使用tf.image.rgb_to_grayscale對影像進行灰階處理：
![image](https://github.com/user-attachments/assets/2262ec6f-57eb-4e73-9ed6-9565ecb4f03f)
![image](https://github.com/user-attachments/assets/2c8b701e-4754-4fad-95af-938bd72c1d8e)

調整影像飽和度
使用tf.image.adjust_saturation，透過提供飽和係數來調整影像飽和度：
![image](https://github.com/user-attachments/assets/b980d3b2-3bfb-4103-aa20-1a65dcf36941)
![image](https://github.com/user-attachments/assets/e9c264ae-9f80-4708-95f6-00a6a5d18a58)

更改影像亮度
使用tf.image.adjust_brightness，透過提供亮度係數來更改圖像的亮度：
![image](https://github.com/user-attachments/assets/26032522-cfd3-43ea-8f7a-48789e4a428e)
![image](https://github.com/user-attachments/assets/4ae11828-4ee0-4b80-afc8-7df20618a5ff)

對影像進行中心裁剪
使用tf.image.central_crop將影像從中心裁切到所需部分：
![image](https://github.com/user-attachments/assets/ab5a19c5-59c8-4032-9869-858fd80bd981)
![image](https://github.com/user-attachments/assets/b98db86e-58e9-43aa-a0b1-023d07573c0f)

旋轉影像
使用tf.image.rot90將影像旋轉90 度：
![image](https://github.com/user-attachments/assets/5ca47bed-ed09-4640-85c2-32e2620a86ab)
![image](https://github.com/user-attachments/assets/ceb13539-67b5-444f-8896-cdb448ab38be)

隨機變換
警告：有兩組隨機圖像運算：tf.image.random*和tf.image.stateless_random*。強烈不建議使用tf.image.random*運算，因為它們使用的是TF 1.x 中的舊RNG。請改用本教學中介紹的隨機圖像運算。有關詳情，請參閱隨機數產生。
對影像應用隨機變換可以進一步幫助泛化和擴展資料集。目前的tf.imageAPI 提供了8 個這樣的隨機影像運算(op)：
tf.image.stateless_random_brightness
tf.image.stateless_random_contrast
tf.image.stateless_random_crop
tf.image.stateless_random_flip_left_right
tf.image.stateless_random_flip_up_down
tf.image.stateless_random_hue
tf.image.stateless_random_jpeg_quality
tf.image.stateless_random_saturation
這些隨機圖像運算純粹是功能性的：輸出僅取決於輸入。這使得它們易於在高性能、確定性的輸入管線中使用。它們要求每一步都輸入一個seed值。給定相同的seed，無論被呼叫多少次，它們都會傳回相同的結果。
註：seed是形狀為(2,)的Tensor，其值為任意整數。

在以下部分中，您將：
回顧使用隨機影像運算來變換影像的範例。
示範如何將隨機變換應用於訓練資料集。

隨機更改影像亮度
透過提供亮度係數和seed，使用tf.image.stateless_random_brightness隨機變更image的亮度。亮度係數在[-max_delta, max_delta)範圍內隨機選擇，並與給定的seed相關聯。
![image](https://github.com/user-attachments/assets/82dfece5-3ae0-4da5-bd51-5ac53fc058b1)
![image](https://github.com/user-attachments/assets/c9725877-72c3-4918-839e-5fba47ced40c)

隨機更改影像對比度
透過提供對比範圍和seed，使用tf.image.stateless_random_contrast隨機變更image的對比。對比範圍在區間[lower, upper]中隨機選擇，並與給定的seed相關聯。
![image](https://github.com/user-attachments/assets/39d876a4-0068-49b2-a30d-ddb3a2c4951c)
![image](https://github.com/user-attachments/assets/d2c526aa-8d7a-4e73-824c-355839ac03f1)

隨機裁剪影像
透過提供目標size和seed，使用tf.image.stateless_random_crop隨機裁剪image。從image中裁剪出來的部分位於隨機選擇的偏移處，並與給定的seed相關聯。
![image](https://github.com/user-attachments/assets/06d661be-d6e1-4720-9790-0fc5f595e9ca)
![image](https://github.com/user-attachments/assets/8551bb31-423b-4bc2-a0b8-8e4b3a1c8098)

對資料集應用增強
我們首先再次下載圖像資料集，以防它們在先前的部分中被修改
![image](https://github.com/user-attachments/assets/7a17dfbc-e39f-4b6b-854c-aa45f1d55db8)
接下來，定義一個用於調整影像大小和重新縮放影像的效用函數。此函數將用於統一資料集中影像的大小和比例：
![image](https://github.com/user-attachments/assets/8e563b2f-5052-404b-9ce7-0e6dbf625cf1)
我們同時定義augment函數，該函數可以將隨機變換應用於影像。此函數將在下一步中用於資料集。
![image](https://github.com/user-attachments/assets/58d84d69-ed26-4ae9-b9ca-ad8cbb776bff)

選項1：使用tf.data.experimental.Counter
建立一個tf.data.experimental.Counter()物件（我們稱之為counter)，並使用(counter, counter) Dataset.zip資料集。這將確保資料集中的每個影像都與一個基於counter的唯一值（形狀為(2,)）相關聯，稍後可以將其傳遞到augment函數，作為隨機變換的seed值。
![image](https://github.com/user-attachments/assets/23f83e38-c5ac-445d-b848-c561153acded)
將augment函數映射到訓練資料集：
![image](https://github.com/user-attachments/assets/2f90bfac-bec2-4bea-8774-c1a576bbdaf3)

選項2：使用tf.random.Generator
建立一個具有初始 值seed的tf.random.Generator物件。在同一個生成器物件上呼叫make_seeds函數會始終傳回一個新的、唯一的seed值。
定義一個封裝容器函數：1) 呼叫make_seeds函數；2) 將新產生的seed值傳遞給augment函數進行隨機變換。
注意：tf.random.Generator物件會將RNG 狀態儲存在tf.Variable中，這表示它可以儲存為檢查點或以SavedModel格式儲存。有關詳情，請參閱隨機數產生。
![image](https://github.com/user-attachments/assets/8a435d31-ebdd-48ab-9ff3-c5fecb0769e4)
將封裝容器函數f對應到訓練資料集，並將resize_and_rescale函數對應到驗證集和測試集：
![image](https://github.com/user-attachments/assets/455ed1e4-1626-4aa4-9e7b-4bd3b75239b0)

後續步驟
本教學示範了使用Keras 預處理層和tf.image進行資料增強。
若要了解如何在模型中包含預處理層，請參閱影像分類教學。
您可能也有興趣了解預處理層如何幫助您對文字進行分類，請參閱基本文字分類教學。
您可以在此指南中了解有關tf.data的更多信息，並且可以在這裡了解如何配置輸入流水線以提高性能。















































































