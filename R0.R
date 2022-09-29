#AMS518 
#Final Project
#Chi-Sheng Lo (ID: 114031563)


#install.packages("C:/Aorda/PSG/R/PSG_3.2.zip", repos = NULL, type = "win.binary") 

library(PSG)
library(readxl)
load("D:/Stony Brook AMS/AMS 2021 Fall/AMS 518 Stochastic and Optimization/AMS518 Assignment/AMS518 A2/problem_fc_indtrack1_max_0025_R/problem_fc_indtrack1_max_0025_short.RData")
R0 <- read_excel("D:/Stony Brook AMS/AMS 2021 Fall/AMS 518 Stochastic and Optimization/AMS518 project/AMS518 Final Project submission_Chi-Sheng Lo/TW50 index tracking project/R0.xlsx")

#R0 <- read_excel("R0.xlsx")
KP_R0 <- read_excel("D:/Stony Brook AMS/AMS 2021 Fall/AMS 518 Stochastic and Optimization/AMS518 project/AMS518 Final Project submission_Chi-Sheng Lo/TW50 index tracking project/KP_R0.xlsx")
KSI <- read_excel("D:/Stony Brook AMS/AMS 2021 Fall/AMS 518 Stochastic and Optimization/AMS518 project/AMS518 Final Project submission_Chi-Sheng Lo/TW50 index tracking project/KSI.xlsx")
KBUY <- read_excel("D:/Stony Brook AMS/AMS 2021 Fall/AMS 518 Stochastic and Optimization/AMS518 project/AMS518 Final Project submission_Chi-Sheng Lo/TW50 index tracking project/KBUY.xlsx")

TWA50R <- as.matrix(R0)
kpol<-as.matrix(KP_R0)
ksi<-as.matrix(KSI)
KB<-as.matrix(KBUY)

#which(stock_matrix==NA)

length(problem.list$matrix_inmmax)<-58446
dim(problem.list$matrix_inmmax)<-c(51, 1146)

problem.list$matrix_inmmax<-TWA50R
problem.list$matrix_ksi<-ksi
problem.list$matrix_ksibuy<-KB
problem.list$matrix_ksipol<-kpol

#problem.list[["matrix_ksibuy"]][1:51]<-rep(1000000,51)
#problem.list[["matrix_ksipol"]][2, ][1:51]<-rep(0,51)


#Creates a built-in R function that incorporates the PSG solver
problem.list$problem_statement <- sprintf (
  "
minimize
  max_risk(matrix_inmmax)
Constraint: <=17
  cardn_pos(0.01, matrix_ksi)
Constraint: <= 0
  buyin_pos(0.01, matrix_ksibuy)
Constraint: <=20000000
  linear(matrix_ksi)
  +variable(trcost)
Constraint: <= 76000
  variable(trcost)
Constraint: <= 0
  -variable(trcost)
  +0.01*polynom_abs(matrix_ksipol)
  +100*cardn_pos(0.01, matrix_ksipol)
  +100*cardn_neg(0.01, matrix_ksipol)
Box: >= 0
Solver: precision=7, stages =30
    "
)

results_A1 <- rpsg_solver(problem.list)


#Point problem_1 means allocation by value
#To get to the weight
sum(results_A1[["point_problem_1"]][1:50])
results_A1[["point_problem_1"]][1:50]/sum(results_A1[["point_problem_1"]][1:50])



