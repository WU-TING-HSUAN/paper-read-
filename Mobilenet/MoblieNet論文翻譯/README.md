# MobileNet論文翻譯
## Abstract
作者提出了一個有效率的模型，稱之為MobileNets，適用在mobile及嵌入式視覺應用上，MobileNet由精簡的體系結構為基礎，並使用depthwise separable convolutions去輕量化深層神經網路。我們引用了兩個簡單的全域參數，可以有效的在延遲及準確性之間取得良好的平衡。這些參數根據問題選適當的模型。作者提出大量的實驗數據，在精準度上顯示出強大的效能，並讓其模型對ImageNet分類。
然後作者展示了MobileNets各種的應用，包話目標檢測、分類、人臉識別、大規模的地點定位。

## Introduction
自從AlexNet贏得後，加上ILSVRC2021的推廣，卷積神經網路已經廣泛的應用在電腦視覺領域上，幾乎是到了無所不在的境界，為了獲得更高的準確率，大家都致力於研究更複雜更深入的神經網路，然而在提升這些準確率的同時並不代表也能夠提高運算速度，在機器人、自駕車等現實的應用中，辨識的任務必須在有限的設備上完成。

本文描述了一個有效率的神經網路及一組兩個超參數，以建立非常小、低延遲的模型，可以輕鬆放入mobile和嵌入式視覺應用系統上。

## Prior Work
近年來，人們對於建立小型高性能的神經網路越來越感興趣，而這些方法也可以分為壓縮預訓練網路或直接訓練小型網路。本文提出一個網路架構，專門使用在資源有限制時的小型網路，而MoibleNets主要用於優化延遲，同時也產生小型的神經網路，許多小型網路的論文只注重於規模，而不考慮速度。

MoibleNets主要是引入depthwise separable convolutions來減少前幾層的運算，<font color="red">平坦網路是由完全因數化卷積層所構成的網路，顯示出因數化網路的潛力</font>，在本文之外，分解網路引用類似的分解捲基層的方法來使用，隨後Xception network展示出如何擴展depthwise separable filters以超越Initiation V3，另一個小型網路Squeezenet，他使用bottleneck方法來設計一個非常小的網路，其他簡化方法包括結構化轉換網路及深度轉換網路。

還有一種獲得小網路的方法為收縮、分解、壓縮預訓練網路，此外還有人提出了各種因數分解來加速預訓練網路，另一種訓練小網路的方法為蒸餾法。他是使用較大的網路來教導小的網路，另一種新興方法是低bit網路。

## MobileNet Architecture
MobileNet最核心的技術為depthwise separable filters，然後我們描述了MobileNet的網路架構，最後給出了兩個模型縮小參數 width multiplier and resolution multiplier.

### 3.1 Depthwise Separable Convolution
MobileNet模型基於 depthwise separable convolutions，它是一種factorized convolutions的形式，它是把標準卷積分解為depthwise convolution和1 x 1的卷積，這作者稱之為pointwise convolution。對於MobileNets，depthwise convolution對每個輸入通道使用filter，然後pointwise convolutin使用1 x 1的卷積將輸出與depthwise convolution作結合。一個標準的卷積網路同時過濾與組合成一個新的輸出，depthwise convolution分為兩個層，一個用於過濾單獨的層與用於組合的單獨層，這種因式分解具有顯著的減少計算量和模型大小的效果。

Figure 2 shows how a standard convolution 2(a) is factorized into a depthwise convolution 2(b) and a 1 × 1 pointwise convolution 2(c).

![](https://i.imgur.com/qjcpqCO.png)

一個標準的卷積層將DF x DF x M，feature map F視為輸入，產生DG x DG x N的feature map G，M是輸入channel數量，DG是正方形輸入feature map的寬和高，N是輸出通道數。

標準卷積層是由kernel數K、大小DK x DK x M x N來參數化，DK假設為正方形的kernel空間維數(where DK is the spatial dimension of the kernel assumed to be square)，M為輸入通道數量，N維輸出通道數量。

假設步數為1，標準卷積的輸出特徵映射計算如下:

![](https://i.imgur.com/uer9lXk.png)

計算成本為:

![](https://i.imgur.com/wVIuwF7.png)

其中計算成本與輸入通道數量M成倍數關係， 輸出通道數為N，kernel大小為DK x DK和feature map大小為DF x DF。

MobileNets首先使用depthwise separable convolutions來打破輸出通道數量和kernel大小之間相互作用，通過這樣的方法可以將filter和組合分為兩個步驟，從而大大的減少計算成本。

Depthwise separable convolution由兩層所組成，depthwise convolutions and pointwise convolutions。我們使用depthwise convolutions來作為為每個輸入通道的filter，Pointwise convolution一個簡單的1 x 1卷積，用於創建深度層輸出的線性組合。

Depthwise convolution可以寫成:

![](https://i.imgur.com/13d1pJ0.png)

其中K是大小為DK x DK x M的深度卷積kernel，其中K中的第M個濾波器用於F中第M個通道已產生輸出特徵G第M個通道。

計算成本為:

![](https://i.imgur.com/zewhOYj.png)

Depthwise convolution對於標準卷積是非常有效的，然而它只過濾數入的channel，它並不會結合他們來創造新的feature，所以為了產生新的feature，需要額外的一層，該層通過1 x 1卷積計算深度卷積輸出的線性組合，Depthwise convolution與 1 × 1(pointwise) convolution結合我們稱之為depthwise separable convolution

depthwise separable convolution花費為:

![](https://i.imgur.com/GXLOWCV.png)

它是和1 x 1卷積計算量的總和

通過將卷積表示為兩步驟濾波和組合的過程，我們可以減少計算量:

![](https://i.imgur.com/KEujpk2.png)


MobileNet使用3×3深方向可分離卷積，其計算量比標準卷積少8到9倍，精度僅略有降低。

### Network Structure and Training
MobileNets 的架構建立於上述所說的depthwise separable convolutions，而MobileNet的結構定義如table1.

![](https://i.imgur.com/0YAGju0.png)

後面所有的層都非線性的所以使用Batchnorm和ReLU，但最後的全連接層除外，因為該層並非是非線性的所以使用<font color="red">softmax</font>層來進行分類

Figure 3對比了規律性的卷積

![](https://i.imgur.com/XD3VCpI.png)

在全連接層之間，將平均池化層的空間解析度降到1，而最後MobileNet總共有28層。

僅用少量的Mult加法來定義網路是不夠的，同樣的是要確保這樣的操作是能夠被有效的實現，譬如說，非結構化的疏鬆矩陣的運算速度通常都不會比密集矩陣來塊，除非疏鬆度非常的高，我們的模型幾乎在所有的計算放入了密集的1 x 1的卷積，這可以通過高度優化的通用矩陣乘( highly optimized general matrix multiply (GEMM))函數來實現，卷積通常都採用GEMM來實現，但需要再記憶體中進行一個名為im2col的初始重新排列，以便將其映射至GEMM.例如這種方法就非常常用在caffe包中使用，而1 x 1卷積並不需要在記憶體中重新排列，可以直接使用GEMM來實現，GEMM是最優化的數值線性代數之一，MobileNet將95%的計算時間花在1 x 1的卷積上，其中有75%花在參數上，如table 2.

![](https://i.imgur.com/rANxIwG.png)

MobileNet模型在TensorFlow中使用RMSprop進行訓練，然而與大模型相反，由於小型模型比較不容易出現overfitting的現象，所以我們並沒有使用非常多的正規化。在訓練MobileNets的時候，我不使用side head 或 label smoothing，通過限制croping的尺寸來減少圖片扭曲。另外，我們發現重要的是在depthwise濾波器上放置很少或沒有權重衰減（L2 正則化），因為它們參數很少，下一節ImageNet中，所有的模型通過同樣的參數進行訓練，但不考慮模型的大小。

### Width Multiplier: Thinner Models
儘管MobileNet的架構已經很小，延遲也很低，但還不夠，有時候需要更快更小的模型，為了達到這樣的境界我們引入了一個非常簡單的參數，稱之為Width Multiplier，他的作用是在每一層均勻的細化網路，輸入通道變為aM，輸出通道變為aN，所以式子會變成

![](https://i.imgur.com/HDJh5mq.png)

其中𝛼屬於0~1之間，通常為1、0.75、0.5、0.25，𝛼等於1的時候為標準的MB網路，𝛼小於1的時候為輕量化的MN網路，有效的減少計算複雜度和參數的數量(大約𝛼的二次方)， Width Multiplier可以有效的利用在任何的模型上，定義一個準確度合理且較輕巧的模型，由於他定義了新的簡化結構，所以並須重新訓練模型。

### Resolution Multiplier: Reduced Representation

降低神經網路計算量的第二個超參數是分辨率乘數ρ，我們使用在輸入影像，在一層上的內部特徵減去相同的乘數，實際上作者透過解析度來影式設定ρ，而最後的計算時間如下圖

![](https://i.imgur.com/v5i5Rzs.png)

ρ通常屬於0到1之間，而輸入得解析度通常在224、192、160、128，ρ等於1的時候為標準的MB網路，ρ小於1的時候為輕量化的MN網路，有效的減少計算複雜度和參數的數量(大約ρ的二次方)

舉例來說，我們可以看到MobileNat中的一個經典的層，看看depthwise separable convolutions,上述所說的兩個參數降低了運算量及參數量，Table3顯示了一個層的運算量以及參數量，第一行顯示出捲基層的計算量以及參數量，輸入特徵圖為14 x 14 x 512，kernel大小為3 x 3 x 512 x512，而我們將在下一個章節討論這個網路的準確性。

## Experiments
實驗結果作者比較著重於depthwise convolution跟上述所說的兩個超參數，來跟其他模型做對比。

### Model Choices
首先，我們展示了與具有完整卷積的模型互相比較，depthwise separable convolutions的MobileNets的結果在Table4.我們可以看到作者的方法在ImageNet資料集上，準確度只低了1%，但在計算量及參數量，都是在數倍以下。

![](https://i.imgur.com/6dcmrvz.png)

接下來我們使用寬度因子的網路，與較少層的網路做比較，為了使MobileNet變得更淺，Table1.中的14 x 14 x512的層數被刪除，比較結果如Table5.在相圖的計算和參數量下，MobileNet的表現比較少層的淺層網路好了3%

![](https://i.imgur.com/jfxvsrw.png)


### Model Shrinking Hyperparameters
Table6.展現出了準確率，使用寬度因子縮小了MobileNet的架構的準確度，計算在不同的𝛼下，準確度會平穩的下降，直到架構縮小到𝛼=0.25為止

![](https://i.imgur.com/RxOM0Y8.png)

Table7.展現在訓練輸入不同的解析度所獲得解析度因子的準度，可以看見在解析度下降的同時，準確率也會下降

Figure4.顯示了由𝛼[1 0.75 0.5 0.25]ρ[224,192,160,128]叉積的16個模型的精準度以及計算量，結果呈現對數型的成長，
![](https://i.imgur.com/5Q8Hyqe.png)


![](https://i.imgur.com/WbEu7Vg.png)

Table8.將完整的Mobilenet與GoogleNet和VGG16進行了比較，我們可以發現MB的準確度幾乎跟VGG16一樣，但小了將近32倍，計算的複雜度低了27倍，也比GOOGLENET更準確，體積更小，計算量減少了在2.5倍以上

![](https://i.imgur.com/iWwJ6a6.png)

Table9.是將輕量化的mb與AlexNet做比較，輕化量的mb準確率比AlexNet高出了4%，參數量差了45倍，計算量要減少了9倍，在參數量相近的Squeezenet上，準確率高出了2.7%，計算量更是差了22倍。

### Fine Grained Recognition
我們在Standford Dogs訓練及上訓練，對mb進行Fine Grained Recognition，與Inception V3做比較，並在網路上找尋大量有雜訊的狗資料來訓練模型，並在Standford Dogs上進行調整，結果如Table10.

![](https://i.imgur.com/uOlKl2F.png)

可以看到mb可以達到將近相同的準確度，而運算量及參數量都大幅的減少。

### Large Scale Geolocalizaton
PlaNet是做大規模地理分類任務，該任務是確認照片會在哪裡的分類，該方法將地球劃分為目標類別的地理網路，並在數百萬張帶有地理標籤的照片上訓練，PlaNet已被證明可以成功地定位各種照片，並勝過Im2GPS

作者使用了MB的框架來重新設計PlaNet，PlaNet本身含有5200萬個參數和574億的計算量，而mbp只有1300萬的參數及50萬的計算量，相比之下效能只是稍微受損，但效果還是比im2gps好很多。

![](https://i.imgur.com/pX3dmVc.png)

### Object Detection
在Table13.中，在faster-rcnn和ssd的框架下，將MobileNet與VGG和Inception V2進行了比較
在我們的實驗中，對SSD的輸入解析度為300，並將Faster-RCNN與輸入解析度分別為300和600，進行比較。

![](https://i.imgur.com/deXiQtB.png)

### 人臉識別
FaceNet是現在最先進的人臉識別模型，為了建立mobile FaceNet，我們通過使用蒸餾法來最小化FaceNet和nb在訓練資料上的輸出平方差來進行訓練，結果如下。

![](https://i.imgur.com/md1KzpA.png)

