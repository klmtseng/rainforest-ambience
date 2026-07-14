# 雷雨環境音景：合成、氣象互動與混音研究調查
撰寫日期：2026-07-14
任務：深山雨夜音景專案 — 雷聲合成 + 雷雨共變 + 混音策略

---

## 一、雷聲物理式／程序式合成

### 1.1 學術論文（核實 URL）

**[P1] Ribner & Roy (1982) — 經典 N-wave 物理模型（未查證全文 PDF，但引用鏈高度一致）**
- 標題：*Acoustics of thunder: A quasilinear model for tortuous lightning*
- 期刊：Journal of the Acoustical Society of America, 72(6), pp. 1911–1925
- 方法：把閃電通道模型化為一條曲折路徑，沿路徑分佈點聲源，每個點源發射弱衝擊 N-wave。整段雷聲由各點源到達聽者的時差疊加產生。這是後續幾乎所有雷聲合成方法的物理基礎。
- 可用性：提供核心參數框架（N-wave 持續時間 ≈ 0.5–1 ms；雷聲「隆隆」= 多點源到達時差）。**全文付費牆，具體數值需查 citepaper 二手資料或 Reiss 2021 的引用。**

**[P2] Reiss, Tez, Selfridge (2021) — 雷聲合成比較評估（核實）**
- 標題：*A comparative perceptual evaluation of thunder synthesis techniques*
- 發表：150th AES Convention, May 2021（AES eBrief 640）
- URL（repo）：https://github.com/joshreiss/thunder-simulation-evaluation（MIT License，已驗證存在）
- 評估對象：5 個模型（Blanco、Brookes/Max MSP、Farnell/PureData、Saksela、Selfridge/Nemisindo）vs. 真實錄音，50+ 人聽測。
- 主要結論：**所有模型都和真實雷聲有顯著差距**；signal-based 方法（如 Farnell）稍優於純物理模型；所有現有模型只產生 mono，且對閃電通道的幾何建模過於簡化。
- 可用性：repo 含 5 個模型原始碼（Pd / Python）+ 聲音樣本，可直接下載試用。

**[P3] Fineberg, Walters, Reiss (2022) — 進階雷聲合成（核實）**
- 標題：*Advances in Thunder Sound Synthesis*
- arXiv：https://arxiv.org/abs/2204.08026（已驗證）
- 發表：AES 152nd Convention, Spring 2022
- 方法：在現有實作基礎上加入「物理啟發的信號設計元素」模擬環境現象（細節在論文 PDF 內，摘要未完整說明）；50+ 人聽測結果：比之前模型更自然，但仍可與真實錄音區分。
- GitHub（互動 demo + 前端）：https://github.com/bineferg/thunder-synthesis（JS/HTML/Python，已驗證）
- 線上體驗：nemisindo.com
- 可用性：論文方法 PDF 可從 arXiv 直接下載。

**[P4] Andy Farnell — *Designing Sound* (MIT Press, 2010) — Thunder 章節**
- 出版社頁面（核實）：https://mitpress.mit.edu/9780262288835/designing-sound/
- 方法：Pure Data 程序式音效，從物理第一原理出發。Thunder 實作在 Ch.~62（Farnell 模型在 P2 評估中表現屬「signal-based 中等」）。
- 可用性：書本付費，但 Pd patch 原始碼可在 https://aspress.co.uk/sd/ 免費下載。

### 1.2 開源實作整理

| Repo | 方法 | 語言 | 授權 |
|---|---|---|---|
| [joshreiss/thunder-simulation-evaluation](https://github.com/joshreiss/thunder-simulation-evaluation) | 5 模型比較（含 Farnell Pd, Saksela, Blanco 物理） | Python + Pd | MIT |
| [bineferg/thunder-synthesis](https://github.com/bineferg/thunder-synthesis) | 物理啟發信號設計（P3 論文配套） | JS + Python | 未確認 |
| Farnell patches | 程序式 Pure Data，含 thunder | Pure Data | 見 aspress.co.uk |

**額外候選（未完整核實）**：
- Saksela 模型（Kai Saksela）——在 P2 repo 中被引用，方法細節標「未查證原始論文」。
- Blanco 模型（物理數位雷聲）——P2 repo 有實作，原始論文標「未查證」。

### 1.3 對我們專案的可用性

- **最快上手**：joshreiss repo 中的 Farnell Pd patch — 轉成 Python/scipy 的路徑最短（filtered noise + layered envelopes）。
- **最高天花板**：P3 論文方法（Fineberg 2022）+ bineferg repo，但需要讀懂完整 PDF。
- **核心物理概念可抄**：N-wave（閃電通道 → 點聲源序列 → 各點到達時差 → 「隆隆」）；實作用 numpy：沿隨機曲折路徑採樣若干點，每點發射帶 envelope 的短脈衝，疊加後加混響。

---

## 二、雷擊後降雨增強（Rain Gush）的氣象觀測數據

### 2.1 核實論文

**[M1] Jayaratne & Saunders (1984) — 「rain gush」現象命名論文**
- 標題：*The "rain gush", lightning, and the lower positive charge center in thunderstorms*
- 期刊：Journal of Geophysical Research, 89(D7), pp. 11816–11818
- URL：https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD089iD07p11816（存在確認，全文需訂閱）
- ADS 鏡像：https://ui.adsabs.harvard.edu/abs/1984JGR....8911816J/abstract（存在確認）
- 機制：在雷擊後，地面雨強有一個短暫激增。論文將此與雷雨胞中下部正電荷層（lower positive charge center）的形成連結，認為是冰晶碰撞帶電後因放電導致湍流消失、大冰雹突然落下所致。
- **量化數據（從多個引用文獻的一致描述提取）**：雷擊後地面降雨激增的時間延遲為 **2–4 分鐘**。

**[M2] Chin et al. (2018) — Ionic wind 機制的 rain gush 模型**
- 標題：*A new approach for rain gush formation associated with ionic wind*
- arXiv：https://arxiv.org/abs/1808.08011（存在確認）
- 機制：閃電中和電荷 → 離子風（ionic wind）消失 → 雲內對流湍流減弱 → 大粒子（rimers）突然下落 = rain gush。
- 量化數據：**論文只提出定性機制，無具體時間延遲數值**。

**[M3] 近期觀測研究（多篇，via ScienceDirect，全文付費）**
- 閃電對雨滴粒徑分佈的影響（2021，Atmospheric Research）：URL https://www.sciencedirect.com/science/article/abs/pii/S0169809521002155
  - 從摘要層級確認：雷擊後地面雨強激增時間延遲 **2–4 分鐘**；droplet 碰撞聚合（collision-coalescence）被增強，雨滴粒徑分佈（RDSD）在閃電後顯著展寬。
- 閃電與雨活動的相關性（2000，Atmospheric Research）：URL https://www.sciencedirect.com/science/article/abs/pii/S0169809500000867
  - 閃電頻率與地面雨率的相關性良好；平均時滯 **4 分鐘**。

### 2.2 跨文獻一致的量化參數（可直接用作合成參數）

| 觀測指標 | 數值範圍 | 來源 |
|---|---|---|
| 雷擊→地面雨強激增時間延遲 | **2–4 分鐘**（平均 ~4 min） | M1 + M3 多篇一致 |
| 閃電頻率→雨率時滯（頻率統計，非單次） | **4–10 分鐘**（更長窗口） | 搜尋摘要 |
| 增強幅度（droplet mass） | 閃電後 30 秒內部分液滴質量增加 **最高 100 倍**（靜電沉降效應）；地面可觀測雨率激增幅度文獻未給統一數字 | M3 引用 |
| 現象類型 | 「transient amplification」非持續增強；持續時間估計數分鐘 | M1, M2, M3 |

**重要判斷**：「雷後雨漸增才自然」這個使用者直覺有真實氣象文獻支持。延遲 2–4 分鐘是 **雲內到地面的物理傳播 + 液滴增長過程**，不是感知誤差。雲內效應（閃電後 30 秒內）遠快於地面可觀測效果。

### 2.3 對合成參數的直接建議

- **ambient rain intensity ramp 觸發延遲**：雷擊事件發生後 **90–240 秒**（1.5–4 分鐘），開始上升。
- **上升坡度**：建議指數曲線（符合液滴增長物理），30–60 秒爬升到峰值。
- **峰值持續**：1–3 分鐘後漸退（現象為 transient，非持久）。
- **縮放版本（for 短片段 / 遊戲音效）**：若不需寫實只需感知自然，可壓縮到延遲 15–40 秒，仍比「立即」更自然且有文獻支持方向性。

---

## 三、環境音混音：Ducking / 響度正規化對自然感的影響

### 3.1 核心問題：為什麼自適應響度正規化毀掉環境音

**EBU R128 / adaptive loudness normalization 的設計假設**：
- 設計給廣播節目的「節目響度」（programme loudness）平衡，目標是讓不同節目間主觀響度一致。
- gating 演算法：在靜音段（低於 -70 LUFS）不計算；對「平均程式響度」做積分。
- 問題：環境音景（ambience）的特徵恰好是**高動態範圍 + 靜音段有意義**——雨聲平均響度低、但雷聲峰值高；若用 EBU R128 target 正規化，系統會把雷聲壓低（拉到 -23 LUFS 目標），同時把靜音段前的細雨放大，結果：所有聲音變成差不多響、失去自然的響度層級感。
- 補充：EBU R128 的 LRA（Loudness Range）指標可描述動態幅度，但「描述」不等於「保護」——串流平台（Spotify, YouTube）的實際正規化實作通常是「輸出整體往 target 移動」，不管 LRA。

**學術/實務佐證（核實）**：
- 論文 *Applying the EBU R128 Loudness Standard in live-streaming sound installations* (NIME 2017)：URL https://www.nime.org/proceedings/2017/nime2017_paper0079.pdf（確認存在）。指出 EBU R128 在「活的聲音裝置」（sound sculpture）應用時需要「必要的妥協」，不完全適用於非廣播 context。
- 一般性原則（engineering consensus）：ambient bed 的動態範圍（70 dB 的雨 → 90 dB 的雷）是自然感的主成分；任何把這個範圍壓縮到 < 10 LU 的正規化都會破壞自然感。

### 3.2 Sidechain Ducking 對環境音的問題

**Wwise HDR 系統的相關行為（核實）**：
- 來源：https://designingsound.org/2013/06/21/finding-your-way-with-high-dynamic-range-audio-in-wwise/（確認存在）
- HDR 系統會在響亮聲音出現時自動壓低其他聲音——這對 SFX vs. ambient 的層級管理有用，但若 thunder（響亮）被設為會 duck rain（ambient），則每次雷聲都讓雨聲消失，效果反自然。
- Wwise 解法：對 ambient bed 聲音設定 `Active Range = 0`，使其**不主動 duck 其他聲音**，也可設定其不被 duck（在 HDR Bus 設定中控制）。

**通用混音原則（engineering consensus，非單一論文）**：
- 「ambient bed 是背景底色」原則：ambient bed 不應對 foreground event（包括 thunder SFX）施加 sidechain ducking，也不應被 sidechain 壓低——這會產生「泵浦感」（pumping），暴露出人工處理。
- 真實雷雨中，雷聲與雨聲**共存而非互替**：雷是短暫疊加在雨上，雨並不因雷聲而降低（雨的物理過程與聲波無關）。
- 推薦做法：thunder SFX 作為**獨立 one-shot 聲音層**疊加在 ambient rain bed 上；兩者分開 bus，無 sidechain 連接；響度層級靠 mixer 靜態設定（thunder 的 peak 高於 rain 的 RMS 20–30 dB 是合理目標）。

### 3.3 實務 Wwise / FMOD 建議（核實來源）

**Wwise 自動 ducking**：
- 文章（https://nickleegameaudio.wixsite.com/website/post/mixing-in-wwise-so-far，確認存在）指出：Wwise Auto-ducking 等待軌道結束才恢復音量，動態不夠平滑；ambient bed 用 Auto-ducking 時容易有「等待感」。
- 推薦：ambient rain bed 走獨立 Bus，**不接任何 Auto-ducking 或 Sidechain**；thunder one-shot 只走自己的 Bus；如果要對話 duck ambient，只 duck ambient bus，不影響 thunder bus。

**對話/SFX 正確 duck 目標**：若有對話，duck ambient rain 6–12 dB，attack 50–100 ms，release 500 ms–2 s；但 thunder bus 不受影響。

---

## 四、三個子題的「查無」記錄

- **SIGGRAPH/DAFx/ICASSP 的 thunder 專門論文**：搜尋結果顯示這些 venue 有一般性 sound synthesis 論文，但**沒找到針對 thunder N-wave 的 SIGGRAPH/ICASSP 專門 paper**；最相關的 venue 是 AES（P2, P3）和 JASA（P1）。標「查過，未見於 SIGGRAPH/DAFx/ICASSP 主題」。
- **"rain gush" 增強幅度的精確 dB 或 mm/hr 數字**：文獻可查到「droplet mass 最高 100 倍」（靜電效應，雲內 30 秒），但**地面可觀測的雨率增幅（mm/hr）沒有跨文獻一致的 dB 或絕對數字**；全文付費牆阻擋了 M1, M3 的詳細數據表。標「方向確認，量值待原文查證」。
- **GDC talk 專門談 ambient no-ducking**：搜尋未找到專門討論「ambient bed 不 duck」的 GDC talk 標題；結論來自 Wwise 技術文件和 engineering consensus，而非單一 GDC 演講。標「查過，無明確 GDC talk 核實 URL」。

---

## 五、參考文獻清單（附 URL 核實狀態）

| ID | 來源 | URL | 狀態 |
|---|---|---|---|
| P1 | Ribner & Roy 1982, JASA 72(6) | 無公開 PDF | 未查證全文，引用鏈確認 |
| P2 | Reiss, Tez, Selfridge 2021, AES 150th | https://github.com/joshreiss/thunder-simulation-evaluation | 核實（repo 存在）|
| P3 | Fineberg, Walters, Reiss 2022, arXiv | https://arxiv.org/abs/2204.08026 | 核實 |
| P4 | Farnell 2010, Designing Sound | https://mitpress.mit.edu/9780262288835/designing-sound/ | 核實 |
| M1 | Jayaratne & Saunders 1984, JGR | https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD089iD07p11816 | 核實（付費牆）|
| M2 | Chin et al. 2018, arXiv | https://arxiv.org/abs/1808.08011 | 核實 |
| M3a | 2021, Atmos. Research（droplet mod.） | https://www.sciencedirect.com/science/article/abs/pii/S0169809521002155 | 核實（付費牆）|
| M3b | 2000, Atmos. Research（lightning-rain corr.）| https://www.sciencedirect.com/science/article/abs/pii/S0169809500000867 | 核實（付費牆）|
| A1 | NIME 2017, EBU R128 sound sculpture | https://www.nime.org/proceedings/2017/nime2017_paper0079.pdf | 核實 |
| A2 | Wwise HDR article, designingsound.org | https://designingsound.org/2013/06/21/finding-your-way-with-high-dynamic-range-audio-in-wwise/ | 核實 |
| A3 | Wwise mixing article, nickleegameaudio | https://nickleegameaudio.wixsite.com/website/post/mixing-in-wwise-so-far | 核實（URL 存在）|
