

library(readxl)

RegBeta <- read_excel("D:/Stony Brook AMS/AMS 2021 Fall/AMS 518 Stochastic and Optimization/AMS518 project/AMS518 Final Project submission_Chi-Sheng Lo/TW50 index tracking project/RegBeta.xlsx")

ModelReg=lm(TWA50 ~ ., data = RegBeta)
ModelReg