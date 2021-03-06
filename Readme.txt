﻿揭秘深度強化學習（人工智慧機器學習技術叢書）-Demystifying Deep Reinforcement Learning
第1章 深度強化學習概覽 
1.1 什麼是深度強化學習？ 
1.1.1 俯瞰強化學習 
1.1.2 來一杯深度學習 
1.1.3 Hello，深度強化學習 
1.2 深度強化學習的學習策略 
1.3 本書的內容概要 
參考文獻 
第2章 強化學習基礎 
2.1 真相--經典的隱瑪律科夫模型（HMM） 
2.1.1 HMM引例 
2.1.2 模型理解與推導 
2.1.3 隱瑪律科夫應用舉例 
2.2 逢考必過瑪律科夫決策過程（MDP） 
2.2.1 MDP生活化引例 
2.2.2 MDP模型 
2.2.3 MDP模型引例 
2.2.4 模型理解 
2.2.5 探索與利用 
2.2.6 值函數和動作值函數 
2.2.7 基於動態規劃的強化問題求解 
2.3 糟糕，考試不給題庫無模型強化學習 
2.3.1 蒙特卡洛演算法 
2.3.2 時序差分演算法 
2.3.3 非同步強化學習演算法 
2.4 學霸來了--強化學習之模仿學習 
2.4.1 模仿學習（Imitation Learning） 
2.4.2 逆強化學習 
本章總結 
參考 
第3章 深度學習基礎 
3.1 深度學習簡史 
3.1.1 神經網路發展史 
3.1.2 深度學習的分類 
3.1.3 深度學習的應用 
3.1.4 深度學習存在的問題 
3.2 深度學習基礎概念 
3.2.1 深度學習總體感知 
3.2.2 神經網路的基本組成 
3.2.3 深度學習訓練 
3.2.4 梯度下降法 
3.2.5 反向傳播演算法（BP） 
3.3 數據預處理 
3.3.1 主成分分析（PCA） 
3.3.2 獨立成分分析（ICA） 
3.3.3 資料白化處理 
3.4 深度學習硬體基礎 
3.4.1 深度學習硬體基礎 
3.4.2 GPU簡介 
3.4.3 CUDA程式設計 
本章總結 
參考 
第4章 功能神經網路層 
4.1 啟動函數單元 
4.2 池化層Pooling layer 
4.3 參數開關Dropout 
4.4 批量歸一化層（Batch normalization layer） 
4.5 全連接層 
4.6 卷積神經網路 
4.7 全卷積神經網路 
4.8 迴圈（遞迴）神經網路（RNN） 
4.9 深度學習的 
本章總結 
參考 
第5章 卷積神經網路（CNN） 
5.1 卷積神經網路 CNN 基礎 
5.1.1 卷積神經網路的歷史 
5.1.2 卷積神經網路的核心 
5.2 卷積神經網路 CNN 結構 
5.2.1 深度卷積神經網路CNN 
5.2.2 深度卷積神經網路CNN視覺化 
5.3 經典卷積神經網路架構分析 
5.3.1 一切的開始--LeNet 
5.3.2 王者回歸--AlexNet 
5.3.3 起飛的時候--VGG 
5.3.4 致敬經典GoogLeNet 
5.3.5 沒有最深只有更深--ResNet 
5.4 對抗網路 
5.4.1 對抗網路（GAN） 
5.4.2 WGAN 
5.5 RCNN 
5.6 CNN的應用實例 
本章總結 
參考 
第6章 迴圈神經網路（RNN） 
6.1 RNN概覽 
6.2 長期依賴（Long-Term Dependencies）問題 
6.3 LSTM 的變體 
本章總結 
參考 
第7章：如何寫自己的CNNC語言實現深度學習 
7.1 如何寫自己的CMake文件 
7.2 如何寫自己神經網路 
7.2.1 啟動函數 
7.2.2 池化函數 
7.2.3 全連接層 
7.3 卷積神經網路 
7.3.1 CNN網路的構建 
7.3.2 CNN前向傳播 
7.3.3 CNN的反向傳播 
7.4 文件解析 
本章總結 
第8章 深度強化學習 
8.1 初識深度強化學習 
8.1.1 深度強化學習概覽 
8.1.2 記憶重播（Memory-Replay）機制 
8.1.3 蒙特卡羅搜尋樹 
8.2 深度強化學習（DRL）中的值函數演算法 
8.2.1 DRL中值函數的作用 
8.2.2 DRL中值函數理論推導 
8.3 深度強化學習中的策略梯度（Policy Gradient） 
8.3.1 策略梯度的作用和優勢 
8.3.2 策略梯度的理論推導 
8.3.3 REINFORCE演算法 
8.3.4 策略梯度的優化演算法 
8.3.5 策略子－評判演算法（Actor-Critic） 
8.4 深度強化學習網路結構 
參考 
第9章 深度強化學習演算法框架 
9.1 深度Q學習 
9.2 雙Q學習 
9.3 非同步深度強化學習 
9.4 非同步優越性策略子-評價演算法 
9.5 DDPG 演算法： 
9.6 值反覆運算網路 
本章總結 
參考 
第10章 深度強化學習應用實例 
10.1 Flappy Bird 應用 
10.2 Play Pong 應用 
10.3 深度地形-自我調整應用（Deep Terrain-adaptive應用） 
10.4 AlphaGo 254 
10.4.1 獨立演算法的研究部分 
10.4.2 AlphaGo演算法 
本章總結 
參考 
附錄： 常用的深度學習框架 
F.1. 穀歌TensorFlow 
F.1.1 TensorFlow 簡介 
F.1.2 TensorFlow 基礎 
F.2 羽量級MXNet 
F.2.1 MXnet介紹 
F.2.2 MXnet基礎 
F.3 來至UCLA 的Caffe 
F.3.1 Caffe 簡介 
F3.2 Caffe基礎 
F.4 悠久的 Theano 
F.4.1 Theano簡介 
F.4.2 Theano基礎 
F.5 30s 入門的Keras 
參考
