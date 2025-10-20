# Google Colab使用教學(附帶MINST案例)
**Google Colab 使用教程<img src="">**

**Google Colab是什麼?**  
Google Colab是Google免費提供的Jupyter筆電環境,支援CPU、GPU和NPU處理,提供諸如 TensorFlow、pytorch、Kernal等主流深度學習框架的環境。該平台部署在雲端,不影響本地使用,因此再拉比再垃圾的電腦也依舊能夠正常使用。
Colab官網:https://colab.research.google.com/

**為什麼選擇該平台**  
Google Colab為所有的開發者免費提供一定的GPU算力,每個人大約能分到一張特斯拉T4顯卡的算力,該顯卡單精度浮點運算能力大約在2070與1080之間,同時擁有16G顯存,如果自己擁有更好的顯卡(如用著4090的富哥)那用自己的會更好。若自己電腦為3060的,雖然單精度浮點運算能力比T4 強,但出於顯存考量以及自己筆電經常外帶的需求,筆者建議使用該平台會更好。 由於是免費提供的,因此該算力也有限制,即每週最多使用三十小時左右(大概,官方也沒有公佈限額,這是動態資源),同時單次運行不能超過12小時,同時若使用用戶過多的情況下不一定能使用上。
Colab Pro 訂閱者的使用量仍會受到限制,但相比非訂閱者可享有的限額要多出約一倍。Colab Pro+ 訂閱者還可獲得更高的穩定性。

**Google Driver是什麼?**  
Google Driver是Google推出的線上儲存服務,類似百度雲端盤,目前有付費和免費兩種模式,免費用戶可享有15G的空間,付費用戶根據方案最多可享有20TB的空間。 Google Driver: https://drive.google.com/drive/

**為什麼要使用該雲端盤**  
如上文所說,Google Colab是谷歌免費提供的Jupyter筆記本環境,那麼每次關閉該環境,伺服器會自動將之前的所有操作進行清除,若不使用Google Driver,則每次都需要上傳數據集和代碼,大大浪費了時間,因此使用該雲盤,和Colab進行鏈接操作,在使用Colab的時候可以調用網盤的數據

**正式教學**

**筆記本創建**
首先進入Google Driver: https://drive.google.com/drive/
<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/01.png">

點選左上角的新建-更多
<img src="https://github.com/Qi-sun-12/Ha/blob/098fe6ec799c24a7a607957cd1ab262ef833597e/02.png">

這時候你已經可以看到Google Colaboratory,若沒有則點擊“關聯更多應用程式”,搜尋“Colab”,安裝第一個即可
<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/03.png">

進入Colab
<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/04.png">

若直接點選Colab的網址則為該頁面
<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/05.png">

這時你只需要點選左上角的檔案-新筆記本就可以進入相同的頁面
左邊有五個選項,分別為目錄、尋找和替換、變數、Secret(秘鑰)、文件
之後點選程式碼執行程序,然後點選變更執行時間類型,在其中硬體加速器部分選擇GPU保存,
Colab便會配置一個帶有GPU的機器,此時筆記本就創建完成了。
<img src="https://github.com/Qi-sun-12/Ha/blob/8716739ab37ba9b22119af48d57772428856a0d6/07.png">

###雲端硬碟掛載由於我們的資料集以及程式碼檔案都放在了Google雲端碟上,因此還需要對Google 雲端硬碟進行掛載在新建立的筆記本中輸入以下程式碼
<img src="https://github.com/Qi-sun-12/Ha/blob/9b40c2239c53e64ef3fd174e9b30107f53941969/06.png">

運行後便可以獲得該視窗
<img src="https://github.com/Qi-sun-12/Ha/blob/5b23c403ac2121f294c429299da12e0fef1cd9fe/08.png">

之後在一系列的視窗中進行登入Google帳號,同時授權對雲端磁碟檔案的讀取與修改,完成後便可實
現掛載
<img src="https://github.com/Qi-sun-12/Ha/blob/45b2a5246b5b880bd85209ce5b4ea9add6ea237b/09.png">

可以看到的是,我們谷歌硬碟裡面的資料已經放在./gdrive/MyDrive 這個目錄裡面,我們再去呼叫的時候就會十分方便,目前筆者的筆記本檔案是在Colab Notebooks 資料夾裡面

**命令列使用**  
在notebook環境下,你只需要在每一行程式碼前面多加一個「!」(注意是英文的感嘆號),便可以
像Linux系統裡的終端指令操作那樣進行指令的輸入
如使用Is指令,便可以得到目前目錄下的路徑
<img src="https://github.com/Qi-sun-12/Ha/blob/cebb374acb603820947a7ffd98cf3336f5a7f8e4/10.png">

**以MINST手寫數字資料集作為範例進行訓練**

**CPU版**
<img src="https://github.com/Qi-sun-12/Ha/blob/45b2a5246b5b880bd85209ce5b4ea9add6ea237b/11.png">

導入相對應的庫

<img src="https://github.com/Qi-sun-12/Ha/blob/e9d6058b0efbab2054dcb8d54584e4798feb759d/12.png">

神經網路的建立

<img src="https://github.com/Qi-sun-12/Ha/blob/ba891428713d3cfc011b9dc20f1fd24d2fe4497f/13.png">

資料集下載(若沒有則會自動下載,若有則會自動跳過並讀取到相對應的資料)

<img src="https://github.com/Qi-sun-12/Ha/blob/e642d7fc8d60a1708a68e4529b72e3fef2f805ff/14.png">

載入資料集(每十張圖片為一批,並隨機打亂)

<img src="https://github.com/Qi-sun-12/Ha/blob/c8bee0b39c5ef916e276eb64673137962a3c4fd2/15.png">

網路實例化

<img src="https://github.com/Qi-sun-12/Ha/blob/1e3eaff58ccdf5b0c8aa22c17f9c653bd82f5618/16.png">

參數優化、學習率與訓練輪次設定

<img src="https://github.com/Qi-sun-12/Ha/blob/cebf38f2a5deacd53204892edec9896f16f92e01/17.png">

開始訓練

<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/18.png">

模型驗證

<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/19.png">

訓練結果如圖
<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/20.png">


具體的程式碼請參考./notebook/CPU版該文件

**GPU版**
如果你已經參考CPU版的程式碼使其成功跑起來的話,你會留意到一件事:為什麼訓練這麼慢? 這時因為我們使用的是CPU去跑,接下來我們就用GPU去跑

首先將更改運行類型,依序點擊程式碼執行程序-更改運行時類型便可以得到以下窗口
<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/21.png">

切換運行類型後倒入庫的同時讀取設備id
<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/22.png">

神經網路建立

<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/23.png">

資料及下載

<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/24.png">

載入資料集,設定每十張照片為一批,並隨機打亂

<img src="https://github.com/Qi-sun-12/Ha/blob/111ca766fcd7f0cefeeb290f5d16f7df23474220/25.png">

網路實例化

<img src="https://github.com/Qi-sun-12/Ha/blob/3294ff8711af14602d65358c76c2908a2bac6fa3/26.png">

優化器、學習率、輪次設定

<img src="">
開始訓練

<img src="">

模型測試

<img src="">

訓練結果如圖
<img src="">
<img src="">
