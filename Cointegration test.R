library(urca)
library(readxl)
volcompare <- read_excel("volcompare.xlsx")
View(volcompare)
vix=volcompare$VIX
sv=volcompare$HTVSM
Time=volcompare$Date
corr=cor(vix, sv, method= c("pearson"))
corr
plot(vix,sv, main="Scatterplot", xlab="VIX", ylab="HTVSM ", pch=19)
abline(lm(vix~sv), col="red") # regression line (y~x)
lines(lowess(vix,sv), col="blue") # lowess line (x,y)
#Johansen cointegration
jotest=ca.jo(data.frame(vix, sv), type="trace", K=2, ecdet="none", spec="longrun")
summary(jotest)
#We can reject the null hypothesis and conclude that there is no cointegration.

#https://www.quantstart.com/articles/Johansen-Test-for-Cointegrating-Time-Series-Analysis-in-R/