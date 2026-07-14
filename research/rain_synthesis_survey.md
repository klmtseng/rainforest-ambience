# Rain Sound Synthesis: Statistical Methods & Open Source Survey
**Date:** 2026-07-13 | **Scope:** Statistical texture matching, procedural DSP, neural (CPU-feasible), real audio datasets

---

## 方向 1：統計式聲音紋理合成（McDermott & Simoncelli 系列）

### 核心論文（核實存在）

**McDermott & Simoncelli (2011)**
- 標題：*Sound Texture Perception via Statistics of the Auditory Periphery: Evidence from Sound Synthesis*
- 發表：Neuron 71, 926–940
- URL（核實）：https://www.sciencedirect.com/science/article/pii/S0896627311005629
- PubMed（核實）：https://pmc.ncbi.nlm.nih.gov/articles/PMC4143345/
- **方法摘要**：用 ERB 濾波器組（30 個耳蝸濾波器）分解信號 → 提取包絡的邊際統計量（mean, variance, skewness, **kurtosis**）、包絡間 Pearson 相關（跨頻帶）、調變功率譜（modulation power spectrum）、調變頻帶間相關；共 ~1500 個統計係數 → 用 L-BFGS 梯度下降把白噪音的統計量逼近目標錄音 → 輸出合成音。
- **直接對應我們的問題**：kurtosis 是其核心統計量之一；頻譜斜率可透過「跨頻帶能量分佈」捕捉。

### 開源實作（全部核實）

#### 1. 原始 MATLAB 工具箱（McDermott Lab 官方）
- GitHub：https://github.com/hackerekcah/Sound_Texture_Synthesis_Toolbox（mirror，核實存在，5 stars）
- 原始來源：McDermott Lab，v1.7（2017 年最後更新）
- 語言：MATLAB 100%
- 實作完整度：包含分析（量測統計量）+ 合成（迭代逼近）全流程
- **可用性**：需要 MATLAB 授權；無直接 Python 等效；但邏輯可逐步移植

#### 2. Python 分析移植（wil-j-wil/texture_stats）
- GitHub：https://github.com/wil-j-wil/texture_stats（核實存在）
- 語言：Python（含 filterbanks.py、texture_stats.py、compare_stats_script.py）
- 授權：MIT
- **注意**：此 repo 只有「分析/量測統計量」功能，**沒有合成（imposing stats on noise）**
- 依賴：Python，推測 numpy/scipy（未確認 requirements.txt 內容）
- **可用性**：可作為量測我們錄音的 kurtosis/modulation spectrum 的基礎；CPU 可跑

#### 3. 統計驅動可微分合成（2025 DAFx 論文 + 開源 PyTorch）
- arXiv（核實）：https://arxiv.org/abs/2506.04073
- 作者：Esteban Gutiérrez, Frederic Font, Xavier Serra, Lonce Wyse（DAFx 2025）
- 標題：*A Statistics-Driven Differentiable Approach for Sound Texture Synthesis and Analysis*
- 專案頁（核實）：https://cordutie.github.io/ddsp_textures/
- GitHub repos（頁面核實存在）：
  - TexDSP：https://github.com/cordutie/ddsp_textures
  - TexStat：https://github.com/cordutie/texstat
- 授權：MIT
- **方法摘要**：TexStat = 統計損失函數（包絡 kurtosis 含於「normalized moments」第四矩中 + 跨頻帶相關 + 調變能量分佈 + 調變頻帶間相關）；TexEnv = 把振幅包絡套在濾波後白噪音上（=程序合成）；TexDSP = DDSP-inspired 模型
- **CPU 可用性**：論文測試用 RTX 4090（forward pass 93.5ms）；TexEnv 本身僅做 IFFT + 元素乘法，**CPU 應可跑**；PyTorch 依賴意謂無 AVX2 的 i5-3210M 可能偏慢，需實測
- **最值得移植**：TexStat 的損失函數直接對應我們的反向最佳化需求

#### 4. 音頻紋理散射矩（Bruna & Mallat 2013）
- arXiv（核實）：https://arxiv.org/abs/1311.0407
- 作者：Joan Bruna, Stéphane Mallat
- **方法摘要**：用迭代小波濾波器組 + 振幅包絡計算散射係數，作為比 MFCC 更豐富的統計描述子，用梯度下降合成；係數數量遠少於 McDermott，計算較省
- 開源：arXiv 頁無直接 repo 連結（查無）；散射變換的 Python 實作可查 Kymatio 庫
- **CPU 可用性**：論文宣稱「比 state-of-the-art 少得多的係數」，傾向 CPU 可跑；待確認

#### 5. 簡化梯度下降文理合成（Nathan Ho，nhthn/texture-resynthesis）
- 文章（核實）：https://nathan.ho.name/posts/texture-resynthesis/
- GitHub（核實）：https://github.com/nhthn/texture-resynthesis（3 stars，8 commits）
- **方法摘要**：STFT magnitude spectrogram → 提取特徵向量（含調變譜概念）→ 梯度下降優化隨機初始頻譜 → Griffin-Lim 相位重建
- **CPU 可用性**：✅ 作者明確說「standard laptop CPU，5 秒音頻約 2 分鐘」
- 依賴：PyTorch + torchaudio
- 授權：未指定（repo 小，個人專案）
- **限制**：未明確實作 kurtosis 匹配；調變譜部分是簡化版

---

## 方向 2：程序式雨聲合成

### Andy Farnell《Designing Sound》Practical 15（核實存在）
- 官方頁面（核實）：https://aspress.co.uk/sd/practical15.html
- 書籍：MIT Press（核實在 Google Books）
- **方法摘要**：5 種模型——簡單脈衝模型、拋物線形狀高斯噪音（bandpass 成形）、脈衝包絡模型（離散雨滴）、水滴共振模型、雨打玻璃模型
- **可下載 Pure Data 檔案**（核實）：rain1.pd, cpraingen.pd, cpulse.pd, rain_on_leaf.pd, spikerain.pd, drops.pd, rain_on_water.pd, windywindow.pd, dropsig.pd, glasswindow.pd, gaussianoise.pd
- **CPU 可用性**：✅ Pure Data 是 CPU-only 實時合成，亦可用 libpd 嵌入 Python
- **對我們的問題**：明確設計各種雨滴物理模型；可提取其濾波器參數作為我們合成器的初始參數

### Raindrop-Generator（DAFx 論文實作，核實存在）
- GitHub（核實）：https://github.com/747745124/Raindrop-Generator
- 基於 DAFx 論文：*Computational Real-time Sound Synthesis of Rain*（查無論文原文 URL）
- 語言：C++ / JUCE（VST/AU 插件）
- **方法摘要**：兩階段——乾表面雨滴（指數衰減正弦）+ 濕表面（數值近似）；可調參數：增益、密度（drops/sec）、頻率係數、Noise Level、高/低通濾波
- **CPU 可用性**：✅ 實時 DAW 插件，純 CPU
- 授權：未指定
- **可移植性**：C++ 核心邏輯可移植成 numpy（指數衰減脈衝 + 隨機卜瓦松時間戳）

### 其他 GitHub Rain Repos（部分核實）
- **azuletto/rain_generator**（核實存在）：Python + pygame + numpy/scipy，程序式，含雷聲；方法偏簡單（白噪音為主）
- **slee1005/rain**（核實存在）：純 Python 雨聲生成器，簡單模型
- **azuletto/rain_generator 標題自承「use white noises」**：正是我們使用者回饋「像白噪音」的症結——此類 repo 不適合直接用

---

## 方向 3：arXiv 近年論文（2020 後）

### MTCRNN（核實）
- arXiv（核實）：https://arxiv.org/abs/2011.12596
- 作者：M. Huzaifah, L. Wyse（2020）
- 標題：*MTCRNN: A multi-scale RNN for directed audio texture synthesis*
- **方法摘要**：多尺度 RNN，不同抽象層級的迴圈網路 + 使用者可控條件向量；明確提到雨聲
- **CPU 可用性**：❌ RNN 推論理論上 CPU 可跑，但論文無 CPU benchmark；i5-3210M 上速度待確認；**方法論可參考，不建議直接部署**
- 開源：查無（arXiv 頁未列 repo）

### Sound Model Factory（核實）
- arXiv（核實）：https://arxiv.org/pdf/2206.13085（2022）
- **方法摘要**：整合式生成音頻模型框架，涵蓋多種紋理合成方法
- **CPU 可用性**：未確認；純方法論參考

### RI Spectrogram 紋理合成（核實）
- arXiv（核實）：https://arxiv.org/abs/1910.09497（2019/2020）
- **方法摘要**：用未訓練 2D CNN 的 feature map cross-correlation 匹配頻譜圖紋理
- **CPU 可用性**：未訓練 CNN 推論理論可 CPU 跑；但對 i5-3210M 可能偏慢

### RainGAN（核實）
- Medium 文章（核實）：https://mdl262.medium.com/raingan-synthesized-environmental-audio-35b78a0f7a77
- **方法摘要**：GAN 合成雨聲
- **CPU 可用性**：❌ GAN 推論需 GPU；不適合本硬體

### 統計驅動可微分合成（2025，已在方向 1 列出）

---

## 方向 4：Hugging Face 資料集

### 真實雨聲資料集（優先用於量測目標統計量）

#### ESC-50（核實）
- HuggingFace：https://huggingface.co/datasets/yangwang825/esc50（核實）
- GitHub 原始（核實）：https://github.com/karolpiczak/ESC-50
- **內容**：50 類環境音，包含「Rain」類別，40 筆 5 秒 WAV（44.1kHz 單聲道）
- **授權**：CC BY-NC（非商用）；ESC-10 子集 CC BY
- **可用性**：✅ 最適合當我們的「目標統計量量測資料集」；40 筆真實雨聲可直接跑 McDermott statistics

#### FSD50K（核實）
- HuggingFace（核實）：https://huggingface.co/datasets/Fhrozen/FSD50k
- **內容**：51,197 筆，200 類（AudioSet Ontology），包含 Natural sounds；個別 clip 混 CC0/CC-BY/CC-BY-NC
- **格式**：16-bit PCM WAV，44.1kHz，mono；總計 34.5 GB
- **是否含雨聲**：AudioSet ontology 含 Rain 類，但此 HuggingFace 鏡像未明確確認（查「大概含」但未直接驗）

#### igorriti/ambience-audio（核實）
- HuggingFace（核實）：https://huggingface.co/datasets/igorriti/ambience-audio
- **內容**：5,916 筆 YouTube 環境音，含雨聲條目（"Rainy Night Serenade" 等）
- **授權**：MIT
- **格式**：CSV/Parquet（YouTube ID + caption）；**需自行從 YouTube 下載音頻，非直接音頻檔**
- **可用性**：低（間接）

#### Freesound.org（核實）
- URL：https://freesound.org（核實）
- **內容**：可搜 "rain"，個別授權 CC0/CC-BY；直接提供音頻下載
- **可用性**：✅ 直接搜 "rain" tag，篩 CC0/CC-BY，可取得大量真實雨聲錄音

### 神經合成模型（供參考，非 CPU 可跑）

#### Stable Audio Open 1.0
- HuggingFace（核實從搜尋結果）：https://huggingface.co/stabilityai/stable-audio-open-1.0
- **可用性**：❌ 擴散模型，需 GPU；i5-3210M 不可用

---

## 對抗驗證：關鍵宣稱核查

| 宣稱 | 驗證狀態 | 說明 |
|---|---|---|
| McDermott & Simoncelli 2011 論文存在 | ✅ 核實（兩個 URL 均可查） | ScienceDirect + PMC |
| hackerekcah/Sound_Texture_Synthesis_Toolbox 存在 | ✅ 核實（5 stars, MATLAB） | McDermott Lab 官方 |
| wil-j-wil/texture_stats 為 Python MIT | ✅ 核實 | 但僅分析，無合成 |
| arxiv 2506.04073 存在（2025 DAFx） | ✅ 核實 | Gutiérrez et al. |
| cordutie/ddsp_textures repo 存在 | ✅ 核實（從官方頁面連結） | MIT 授權 |
| aspress.co.uk/sd/practical15.html 存在 | ✅ 核實（含下載 .pd 檔） | Farnell Designing Sound |
| 747745124/Raindrop-Generator 存在 | ✅ 核實（C++/JUCE） | DAFx 論文實作 |
| ESC-50 含 Rain 類 | ✅ 核實（40 clips, CC BY-NC） | karolpiczak/ESC-50 |
| nhthn/texture-resynthesis CPU 可跑 | ✅ 作者本人明確說明（2 分鐘/5 秒） | PyTorch + CPU |
| TexEnv CPU 可跑 | ⚠️ 推測（操作簡單）但論文以 RTX 4090 測試 | 需實測 |
| MTCRNN 有開源 repo | ❌ 查無 | arXiv 頁未列 |
| RainGAN 有 GitHub | ❌ 查無可用 repo（僅 Medium 文章）| — |
| FSD50K 含雨聲 | ⚠️ 大概（AudioSet 含 Rain 類）但未直接確認 | 待確認 |

---

## 建議路線

### 第一選擇：統計匹配合成（最直接解決「像白噪音」問題）

**目標**：用使用者錄音的 kurtosis / 頻譜斜率 / 調變譜作為目標 → 反向調合成參數

**推薦移植路線**：

1. **量測工具**：用 `wil-j-wil/texture_stats`（Python, MIT）量測使用者真實雨聲的統計量，以及 ESC-50 Rain 類別的平均統計量作為驗證基準。
2. **合成核心**：移植 `cordutie/ddsp_textures` 的 TexEnv（IFFT + 包絡套用在 filterbank 白噪音上）+ TexStat 損失函數（kurtosis 含於其中），改成 scipy 梯度下降版（繞開 PyTorch 以減少 AVX2 相依性）。若 PyTorch 在 i5-3210M 跑得動，直接用 TexStat + scipy.optimize.minimize 最佳化合成參數。
3. **初始參數來源**：參考 Farnell Practical 15 的濾波器設計（低頻成分加重）作為初始化，加速收斂。
4. **真實雨聲資料**：從 Freesound.org 下載 CC0/CC-BY Rain 錄音（或直接用 ESC-50 的 40 筆）作為目標，量測統計量後填入最佳化目標。

### 第二選擇（若 PyTorch 在目標機器偏慢）

直接移植 Farnell Practical 15 的「parabolic shaped noise」模型成 numpy：bandpass 噪音 × 余弦包絡 × 隨機卜瓦松雨滴時間戳，再用統計量量測工具驗證 kurtosis 是否接近目標值，手動調整密度/頻率參數。

### 不建議（硬體限制）

- 任何 GAN / Diffusion / MTCRNN 路線——需 GPU，i5-3210M 無法實用
- Stable Audio Open 或其他擴散模型

---

## 查無項目說明

- MTCRNN 開源 repo：用關鍵字 "MTCRNN audio texture github" 查無；arXiv 頁面未列
- RainGAN GitHub repo：僅有 Medium 文章，無公開 code
- Bruna & Mallat (2013) Scattering Moments 官方 repo：查無；建議查 **Kymatio** Python 庫（散射變換的現代實作，CPU 可跑，`pip install kymatio`）作為替代

---

## 主要來源（核實 URL）

- McDermott & Simoncelli 2011（PMC）：https://pmc.ncbi.nlm.nih.gov/articles/PMC4143345/
- 原始 MATLAB 工具箱鏡像：https://github.com/hackerekcah/Sound_Texture_Synthesis_Toolbox
- Python 統計量分析移植：https://github.com/wil-j-wil/texture_stats
- 2025 DAFx 統計驅動合成：https://arxiv.org/abs/2506.04073
- TexDSP repo：https://github.com/cordutie/ddsp_textures
- TexStat repo：https://github.com/cordutie/texstat
- Farnell Practical 15（含 .pd 下載）：https://aspress.co.uk/sd/practical15.html
- Raindrop-Generator（C++/JUCE）：https://github.com/747745124/Raindrop-Generator
- nhthn/texture-resynthesis（CPU, Python）：https://github.com/nhthn/texture-resynthesis
- MTCRNN arXiv（2020）：https://arxiv.org/abs/2011.12596
- ESC-50 Dataset：https://github.com/karolpiczak/ESC-50
- Freesound.org：https://freesound.org
- igorriti/ambience-audio（HuggingFace, MIT）：https://huggingface.co/datasets/igorriti/ambience-audio
