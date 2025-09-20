# 安裝必要套件（只需安裝一次）
install.packages("keras")
install.packages("magick")
install.packages("caret")

# 載入套件
library(keras)
library(magick)
library(caret)

# 安裝 Keras 和 TensorFlow（只需執行一次）
install_keras()

# 圖片大小
img_size <- c(150, 150)

# 設定資料夾路徑
train_dir <- "C:/Users/Win/Desktop/data analyze/final exam/train"
valid_dir <- "C:/Users/Win/Desktop/data analyze/final exam/test"

# 建立資料產生器
train_gen <- image_data_generator(rescale = 1/255)
valid_gen <- image_data_generator(rescale = 1/255)

# 載入訓練與驗證資料
train_data <- flow_images_from_directory(
  directory = train_dir,
  generator = train_gen,
  target_size = img_size,
  batch_size = 32,
  class_mode = "binary"
)

valid_data <- flow_images_from_directory(
  directory = valid_dir,
  generator = valid_gen,
  target_size = img_size,
  batch_size = 32,
  class_mode = "binary"
)

# 建立 CNN 模型
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(150,150,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

# 編譯模型
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(),
  metrics = "accuracy"
)

# 訓練模型
history <- model %>% fit(
  train_data,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = valid_data,
)

# ===============================
# 🧪 測試模型並產生混淆矩陣
# ===============================

# 測試資料夾內圖片路徑
files <- list.files(valid_dir, full.names = TRUE, pattern = "\\.jpg$", recursive = TRUE)

# 你可以改成要測幾張
n <- length(files)

# 建立真實與預測標籤的容器
y_true <- c()
y_pred <- c()

for (file in files[1:n]) {
  # 根據檔名判斷真實標籤
  if (grepl("cat", basename(file), ignore.case = TRUE)) {
    y_true <- c(y_true, 0)
  } else if (grepl("dog", basename(file), ignore.case = TRUE)) {
    y_true <- c(y_true, 1)
  } else {
    next  # 跳過非貓狗圖
  }
  
  # 圖片前處理 + 預測
  img <- image_load(file, target_size = img_size)
  img_array <- image_to_array(img)
  img_array <- array_reshape(img_array, c(1, 150, 150, 3))
  img_array <- img_array / 255
  
  pred <- model %>% predict(img_array)
  y_pred <- c(y_pred, ifelse(pred < 0.5, 0, 1))
}

# 轉成分類標籤（文字）
y_true <- factor(y_true, levels = c(0,1), labels = c("cat", "dog"))
y_pred <- factor(y_pred, levels = c(0,1), labels = c("cat", "dog"))

# 混淆矩陣
conf_mat <- confusionMatrix(y_pred, y_true)
print(conf_mat)
