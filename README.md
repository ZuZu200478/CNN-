使用網路上公開資料:
https://www.kaggle.com/datasets/tongpython/cat-and-dog

分析主題：
貓狗影像分類模型的建立

研究動機：
貓與狗為最常見的寵物，相關影像資料龐大，適合作為訓練與測試影像分類模型

分析目的:  
建立一個CNN模型 自動判斷輸入影像是「貓」或「狗」
探討模型在不同資料上的準確度與能力

業務價值: 降低人工影像分類的人力與時間成本

訓練結果:
Confusion Matrix and statistics

            Reference
  Prediction cat dog
          cat 88 26
          dog 12 74

             Accuracy: 0.81
               95% CI: (0.7487, 0.8619)
  No Information Rate: 0.5
  p-value (Acc > NIR): < 20-16

               Kappa : 0.62

Mcnemar's Test p-value: 0.03496

Sensitivity: 0.8800
Specificity: 0.7400
Pos Pred Value: 0.7719
Neg Pred Value: 0.8605
Prevalence: 0.5000
Detection Rate 0.4400
Detection Prevalence: 0.5700
Balanced Accuracy 0.8100

"Positive class: cat
