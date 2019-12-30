from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix


df_raw = pd.read_csv('bloom_polarity.csv')


df_raw = df_raw.iloc[:,1:]
df_raw = df_raw.iloc[:-1,:] # shifting the open,polarity by one row up and removing last row row
colm = ['Close','Date']
df_raw = df_raw.drop(colm, 1)
cols = df_raw.columns.tolist()
print(cols)


df_raw.dtypes
df_raw = df_raw.convert_objects(convert_numeric=True)
df_raw.isnull().sum().sum()
df_raw = df_raw.fillna(0)
#df_raw.dropna()



#scalery = StandardScaler().fit(df_raw.iloc[:,-1])
#X_train = scalerX.transform(X_train)
#y_train = scalery.transform(y_train)

ymean = np.mean(np.array(df_raw.iloc[:,-1]), axis=0)
ystd = np.std(np.array(df_raw.iloc[:,-1]), axis=0)

df_raw = pd.DataFrame(preprocessing.scale(df_raw)) #gradient clipping

# numpy array
df_raw_array = df_raw.values

target = [df_raw_array[i,:] for i in range(0, len(df_raw))]
# the length of the sequnce for predicting the future value
sequence_length = 20 #hyperparameter

# convert the vector to a 2D matrix
matrix_load = convertSeriesToMatrix(target, sequence_length)

# shift all data by mean
matrix_load = np.array(matrix_load)

print ("Data  shape: ", matrix_load.shape)

# split dataset: 90% for training and 10% for testing
train_row = int(round(0.8 * matrix_load.shape[0]))
train_valid = int(round(0.9 * matrix_load.shape[0]))
train_set = matrix_load[:train_row, :]

# shuffle the training set (but do not shuffle the test set)

#np.random.shuffle(train_set)

# the training set
X1_train = train_set[:, :-1,:]
variables = df_raw.shape[1] - 1
# the last column is the true value to compute the mean-squared-error loss
X2_train = train_set[:, -1,:variables].reshape(X1_train.shape[0],1,variables)
y_train = train_set[:, -1,-1].reshape(X1_train.shape[0],1,1) # last value is target



# Validation set
X1_valid = matrix_load[train_row:train_valid, :-1,:]
X2_valid = matrix_load[train_row:train_valid, -1,:variables].reshape(X1_valid.shape[0],1,variables)
y_valid = matrix_load[train_row:train_valid, -1,-1].reshape(X1_valid.shape[0],1,1)


#Train + Validation set 
X1_tvalid = matrix_load[:train_valid, :-1,:]
X2_tvalid = matrix_load[:train_valid, -1,:variables].reshape(X1_tvalid.shape[0],1,variables)
y_tvalid = matrix_load[:train_valid, -1,-1].reshape(X1_tvalid.shape[0],1,1)


#Test set
X1_test = matrix_load[train_valid:, :-1,:]
X2_test = matrix_load[train_valid:, -1,:variables].reshape(X1_test.shape[0],1,variables)
y_test = matrix_load[train_valid:, -1,-1].reshape(X1_test.shape[0],1,1)


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(output_dim = 1, activation='linear')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)

# decode a one hot encoded string
#def one_hot_decode(encoded_seq):
#	return [argmax(vector) for vector in encoded_seq]

# configure problem
#n_features = 5
#n_steps_in = 20
#n_steps_out = 1
# define model

train, infenc, infdec = define_models(variables+1, variables, 4) #last value is number of layers
train.compile(optimizer='adam', loss='mse', metrics=['mse'])


# generate training dataset
#X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 10000)
#print(X1.shape,X2.shape,y.shape)

# train model
train.fit([X1_train, X2_train], y_train, epochs=100, batch_size= 100)

train.fit([X1_train, X2_train], y_train, epochs=100, batch_size= 50)

#train.fit([X1_train, X2_train], y_train, epochs=150, batch_size= 25)

#train.fit([X1_train, X2_train], y_train, epochs=20, batch_size= 15)

# evaluate the validation result
valid_mse = train.evaluate([X1_valid, X2_valid], y_valid, verbose=1)
print ('\nThe mean squared error (MSE) on the valid data set is %.3f over %d valid samples.' % (valid_mse[0], len(y_valid)))

# get the predicted values
predicted_values = (train.predict([X1_valid, X2_valid]))
num_valid_samples = len(predicted_values)
predicted_values = predicted_values.reshape(num_valid_samples,1)
predicted_values.shape


# plot the results
fig = plt.figure()
plt.plot((y_valid.reshape(y_valid.shape[0],1))[0:200])
plt.plot((predicted_values)[0:200])
plt.ylabel('Close Price')
plt.show()


epochs = 200
batch_size = 50
fig.savefig('batch_size_' + 
            str(batch_size)+'_epochs_'+ str(epochs)+'_valid_mse'+str('%.4f'%(valid_mse[0]))+'.png', bbox_inches='tight')


# save the result into csv file
valid_result = np.vstack((np.array(predicted_values).reshape(1,X1_valid.shape[0])[0], y_valid.reshape(X1_valid.shape[0],))) 
valid_result = pd.DataFrame(valid_result.T)
valid_result[0] = valid_result[0]*ystd + ymean
valid_result[1] = valid_result[1]*ystd + ymean
valid_result.columns = ['predicted', 'actual']

df = valid_result

df["diff"] = pd.rolling_apply(df['actual'], 2, lambda x: x[0] - x[1])
df['A'] = np.where(df['diff']>=0, 1, 0)
df["diff"] = pd.rolling_apply(df['predicted'], 2, lambda x: x[0] - x[1])
df['P'] = np.where(df['diff']>=0, 1, 0)

t = sum(df['A'] == df['P'])/df.shape[0]
valid_true_sum = sum(df['A'] == df['P'])

#plot
fig = plt.figure()
plt.plot((df["predicted"])[0:200])
plt.plot((df["actual"])[0:200])
plt.ylabel('Close Price')
plt.show()
epochs = 400
batch_size = 50
fig.savefig('batch_size_' + 
            str(batch_size)+'_epochs_'+ str(epochs)+'_valid_mse'+str('%.4f'%(valid_mse[0]))+'.png', bbox_inches='tight')


#fit model on train+valid
train, infenc, infdec = define_models(variables+1, variables, 4) #last value is number of layers
train.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])


train.fit([X1_tvalid, X2_tvalid], y_tvalid, epochs=100, batch_size= 100)
train.fit([X1_tvalid, X2_tvalid], y_tvalid, epochs=100, batch_size= 50)


# evaluate the test result
test_mse = train.evaluate([X1_test, X2_test], y_test, verbose=1)
print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))

# get the predicted values
predicted_values = (train.predict([X1_test, X2_test]))
num_test_samples = len(predicted_values)
predicted_values = predicted_values.reshape(num_test_samples,1)
predicted_values.shape


# plot the results
fig = plt.figure()
plt.plot((y_test.reshape(y_test.shape[0],1))[0:280])
plt.plot((predicted_values)[0:280])
plt.ylabel('Close Price')
plt.show()
#fig.savefig('output_close_price_forecasting_100.jpg', bbox_inches='tight')
epochs = 200
batch_size = 50
fig.savefig('batch_size_' + 
            str(batch_size)+'_epochs_'+ str(epochs)+'_test_mse'+str('%.4f'%(test_mse[0]))+'.png', bbox_inches = 'tight')



## save the result into txt file
#test_result = np.vstack((np.array(predicted_values).reshape(1,197)[0], y_test.reshape(X1_test.shape[0],))) 

#output = pd.DataFrame(predicted_values)


# save the result into csv file
test_result = np.vstack((np.array(predicted_values).reshape(1,X1_test.shape[0])[0], y_test.reshape(X1_test.shape[0],))) 
test_result = pd.DataFrame(test_result.T)
test_result[0] = test_result[0]*ystd + ymean
test_result[1] = test_result[1]*ystd + ymean
test_result.columns = ['predicted', 'actual']

test_result.to_csv('batch_size_' + 
                   str(batch_size)+'_epochs_'+ str(epochs)+'_test_mse'+str('%.4f'%(test_mse[0]))+'.csv')
df = test_result

df["diff"] = pd.rolling_apply(df['actual'], 2, lambda x: x[0] - x[1])
df['A'] = np.where(df['diff']>=0, 1, 0)
df["diff"] = pd.rolling_apply(df['predicted'], 2, lambda x: x[0] - x[1])
df['P'] = np.where(df['diff']>=0, 1, 0)

t = sum(df['A'] == df['P'])/df.shape[0]
test_true_sum = sum(df['A'] == df['P'])

test_true_diff_perc = test_true_sum/(test_result.shape[0])


#plot
fig = plt.figure()
plt.plot((df["predicted"])[0:200])
plt.plot((df["actual"])[0:200])
plt.ylabel('Close Price')
plt.show()
epochs = 400
batch_size = 50
fig.savefig('batch_size_' + 
            str(batch_size)+'_epochs_'+ str(epochs)+'_valid_mse'+str('%.4f'%(valid_mse[0]))+'.png', bbox_inches='tight')


#df1 = test_result

#df.loc[-1] = [0,0]  # adding a row
#df.index = df.index + 1  # shifting index
#df = df.sort_index()  # sorting by index
#
#df1["dummy"] =  df['actual']
test_result.shape


