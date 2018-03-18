install.packages("tseries")
install.packages("forecast")
install.packages("rugarch")

library("tseries")
library("forecast")
library("rugarch")

# adf - augmented dicky fuller test to check for stationarity
#n = nrow(price)
retn = price[2:n, 1] - price[1:(n-1), 1]
adf.test(retn)


diff(price)
ret = diff(log(price))*100 # log returns, in the interest of client
plot(ret, main = "Compound Returns")

# ARIMA order
model = auto.arima(ret, ic = "aic", trace = TRUE)
model

Box.test(model$residuals^2, type = "Ljung-Box")
# Ljung-Box test

# whole data is not used for training, but only some part of data is used (moving window)
# window size depends on index, sector, country, market etc.
k = 1
j = 1
mwin = 1:7500
dim(mwin = c(150,50))
for(j in 1:50){
  l = j + 149
  for(i in j:l) {
    mwin[, j] = ret[j:l]
  }
}
mwin


#model = (sGARCH, eGARCH, iGARCH)
# model specifications are defined here
myspec = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),# hit and trial
                    mean.model = list(armaOrder = c(0,1)))# we get from earlier fit
#the above orders are derived using arima test done earlier

ret.forecast = 1:50
dim(ret.forecast) = c(1,50)
for(t in 1:50){
  # fitting the model using specifications defined above
  garmodel = ugarchfit(spec = myspec, data =mwin[,t], likelihood = T)
  # forecasting using the model fitted above, forecasting for one day ahead
  myforecast = ugarchforecast(garmodel, n.ahead = 1)
  a = myforecast@forecast 
  r1= a$seriesFor # get the return value
  sig1 = a$sigmaFor # get the volatility value
  
  ret.forecast[,t] = r1[,1]
}

ret.forecast
z = ret.forecast
y = z[1,]
write.table()



# split data into train and test
index = sample(1:nrow(data), round(0.75*nrow(data)))
train = data[index, ]
test = data[-index, ]
lm.fit = glm(returns~., data = train)
summary(lm.fit)
pr.lm = predict(lm.fit, test)
MSE.lm = (sum(pr.lm-test$returns))^2/nrow(test)


# doubt : why is normalization / scaling required?
# normalization (scaling the errors)
maxs = apply(data, 2, max)
mins = apply(data, 2, min)
scaled = as.data.frame(scale(data, venter =mins, scale = maxs - mins))

train_ = scaled[index, ]
test_ = scaled[-index, ]



# get the errors from garch model and model them using neural nets
# this can be done in python
# fitting neural net
install.packages("neuralnet")
library(neuralnet)
n = names(train_)
f = as.formula(paste("return~", paste(n[!n %in% "returns"], collapse = "")))
nn = neuralnet(f, data = train_, hidden = c(5,2), linear.output = T)
summary(nn)
plot(nn)








