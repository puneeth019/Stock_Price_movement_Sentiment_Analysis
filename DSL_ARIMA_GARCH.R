# RTSM project
# Code for ARIMA modelling

# set working directory

# load packages
library(tseries)
library(forecast)
library(rugarch)
library(TSA) # for "periodogram"


# Load historical prices
data <- read.csv(file = 'DJIA_final.csv', header = TRUE, sep = ",")
head(data) # look at the data

# convert variable "Timestamp" from factor to class-"POSIXct"type
#data$Date <- as.POSIXct(x = as.character(data$Date), format = "%Y-%m-%d")
data <- data$Adj.Close # get Adjusted Closing Price
plot.ts(x = data)


# Augmented Dicky-Fulelr Test to check for stationarity of the data
# If p-value is less than 0.05 reject null hypothesis with 95% confidence
# and vice-versa
adf.test(data)
# p-value is >0.05, hence data is not stationary



#if not stationary, make data stationary by differencing or detrending
#denote n the number of time periods:
#n <- length(data)
#retn <- data[2:n] - data[1:(n-1)]
#adf.test(retn) # perform dicky-fuller test again after differencing
# p-value is <0.05, hence data is stationary




###### Convert index prices into Compund returns #####
# Though the data is already stationary, this is done because this 
# is a requirement from clients
# Also Converting index prices into Compound returns is equivalent to
# Log-differencing (done to detrend the data)
ret <- diff(log(data), differences = 1)*100 # Log-Differencing
adf.test(x = ret) # p-value data is stationary
plot.ts(ret, main = "Compound Returns")


# Examine acf (auto-correlation function aka correlogram) and 
# pacf(partial auto-correlation function aka partial corrleogram)
acf(ret, lag.max = 100, main = "Returns")
pacf(ret, lag.max = 100, main = "Returns")



# Divide the data into Train and Test components
# First 80% data is Train, next 10% data is validation and next 10% is Test
data_train <- data[1:1430] # first 90% of the data
data_validation <- data[1431:1609] # 10% of the data
data_test <- data[1610:1788] # 10% of the data

# Function to get periodicity of the data sorted in the order
get_perodicity <- function(dem){
  p <- periodogram(dem)
  dd <- data.frame(frequency = p$freq, amplitude = p$spec)
  order <- dd[order(-dd$amplitude),]
  order$time <- 1/order$frequency
  order
}

get_perodicity(ret) %>% head(10)
# no seasonality is present in the data



adf.test(ret) # check for stationary using dicky fuller
acf(x = ret, main = "Auto Correlogram", lag.max = 100)
pacf(x = ret, main = "Partial Auto Correlogram", lag.max = 100)
plot(temp, xlab = "Time", ylab = "Residuals", 
     main = "Residuals after removing seasonalities")

# ARIMA modelling 
model <- auto.arima(y = ret, parallel = T, 
                    num.cores = 8, stepwise = F,
                    max.p = 5, max.Q = 5, D = 3, seasonal = T)
model

library(LSTS) # plot results of Ljung box test
Box.Ljung.Test(z = model$residuals, lag = 100, 
               main = "Ljung-Box test on residuals")
Box.test(x = model$residuals, lag = 100, 
         type = "Ljung")

acf(model$residuals, lag.max = 100, main = "ACF plot of residuals")
pacf(model$residuals, lag.max = 100, main = "PACF of residuals")
qqnorm(model$residuals); qqline(model$residuals)
#shapiro.test(model$residuals)
#library(nortest)
#ad.test(model$residuals)
# qq plot of residuals to check for normality

#histogram of residuals
hist(model$residuals, breaks = "FD", xlab = "Residuals", 
     main = "Histogram of residuals", ylim = c(0,200))

# checking if there is need for GARCH model in r
Box.Ljung.Test(model$residuals^2, 
               main = "Ljung-Box test on resuals^2") # hence model volatilities 
Box.test(x = model$residuals^2, lag = 100)

plot.ts(model$residuals^2, main = "Volatility Clusters")  
  #ACF shows ARCH effect present upto more than 30 lags

#Estimation of GARCH parameters
library(rugarch)

## Volatility:
sgarch <- ugarchspec(variance.model = list(model = "sGARCH", 
                                           garchOrder = c(1,1),
                                           submodel = "GARCH"),
                            mean.model = list(armaOrder = c(3,2),
                                              include.mean = T),
                            distribution.model = "sged")
mvol_sgarch <- ugarchfit(sgarch, ret)
mvol_sgarch   #12.323


egarch <- ugarchspec(variance.model = list(model = "eGARCH", 
                                           garchOrder = c(1,1,1),
                                           submodel = "EGARCH"),
                     mean.model = list(armaOrder = c(3,2),
                                       include.mean = T),
                     distribution.model = "sged")
mvol_egarch <- ugarchfit(egarch, ret)
mvol_egarch   #


gjrgarch <- ugarchspec(variance.model = list(model = "gjrGARCH", 
                                           garchOrder = c(1,1,1),
                                           submodel = "GJRGARCH"),
                     mean.model = list(armaOrder = c(3,2),
                                       include.mean = T),
                     distribution.model = "sged")
mvol_gjrgarch <- ugarchfit(gjrgarch, ret)
mvol_gjrgarch   #

##Choose sGARCH(1,1) as best model for returns


# create moving windows for forecasting
window_size <- 1430
# Window Size is 150, this means that past 150 points are used for forecasting
pred_days <- 358 # Forecast for 179(validation) + 179(test) days
mwin <- seq(from = 1, to = window_size * pred_days, by = 1)
dim(mwin) <- c(window_size, pred_days)
end <- length(ret) - window_size + 1
start <- end - pred_days + 1

for(j in start:end){
  l = j + window_size - 1
  
  for(i in j:l){
    k = start - j + 1
    mwin[, k] <- ret[j:l]
    }
  }



#model = (sGARCH, eGARCH, iGARCH)
myspec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1), 
                                           submodel = "GARCH"),
                     mean.model = list(armaOrder = c(3,2), include.mean = T),
                     distribution.model = "sged")
ret.forecast <- 1:pred_days
dim(ret.forecast) = c(1, pred_days)
for(t in 1:358){
  garmodel <- ugarchfit(spec = myspec, data = mwin[, t], likelihood = T)
  myforecast <- ugarchforecast(garmodel, n.ahead = 1)
  a <- myforecast@forecast
  r1 <- a$seriesFor
  #sig1 <- a$sigmaFor
  ret.forecast[, t] <- r1[, 1]
  print(t)
}


#calculate stock prices from compound returns
x <- 10^(ret.forecast[1, ]/100)
price.forecast <- NA # initialize price.forecast
price.forecast[1] <- x[1] * ret[1429]
for(i in 2:358) {
  price.forecast[i] <- x[i] * price.forecast[i-1]
}

plot.ts(price.forecast)
