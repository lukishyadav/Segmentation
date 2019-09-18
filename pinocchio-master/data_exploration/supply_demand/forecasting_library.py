# forecasting library
'''
Purpose:
    Provide a simple interface for comparing forecasting methods for univariate time series data.
    All functions should take a df in the following format:
        index - timeseries data
        0th column - continuous values

Arguments:
    df - dataframe
    train_ratio - if applicable, train/test ratio split
    draw - whether to draw the results as a graph
    path_prefix - where to save the result to and what the prefix of the filename will be

Returns:
    functions should return rmse for the forecasting performed (best if more than one is done)
'''


def rmse(targets, predictions):
    '''
    takes series or df
    make sure to call get 0th index if df
    '''
    return ((targets - predictions) ** 2).mean() ** 0.5


def linear_regression(df, train_ratio=0.67, draw=False, path_prefix=None):
    '''
    args:
        df - dataframe with date as index and count as values
        draw - directory to save plot

    returns:
        draws a plot where directed
        returns y_actual, y_predicted
    '''

    import matplotlib.pyplot as plt
    import numpy
    import os
    from sklearn.linear_model import LinearRegression

    def plot():
        params = '{} predictions, Linear Regression {} to {}'.format(
            test_size,
            df.index.min(),
            df.index.max())
        stats_results = 'rmse: {} RSquared: {}'.format(
            round(rmse(y_test, y_pred)[0], 3),
            round(r_sq, 3))
        plt.title(f'{params}\n{stats_results}')
        plt.plot(x_train, training_line, label='train linear regression')
        plt.plot(x_test, y_pred, label='test linear regression')
        plt.plot(range(0, train_size + test_size), df, label='rentals')
        plt.legend(loc='upper left')
        plt.grid()
        plt.savefig('{}/{}_linear_regression.png'.format(
            draw, str(df.columns[0])))
        plt.close()

    train_size = int(len(df) * 0.67)
    test_size = len(df) - train_size
    y_train = df[0:train_size]
    y_test = df[train_size:train_size+test_size]

    x_train = numpy.array(range(0, train_size)).reshape((-1, 1))
    x_test = numpy.array(
        range(train_size, train_size + test_size)).reshape((-1, 1))

    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    training_line = model.coef_ * x_train + model.intercept_

    r_sq = model.score(x_train, y_train)

    plot() if draw else None
    return y_test, y_pred


def simple_exponential_smoothing(df, train_ratio=0.67, draw=False, path_prefix=None):
    '''
    args:
        df - dataframe with date as index and count as values
        draw - directory to save plot

    returns:
        draws a plot where directed
        returns lowest rmse
    '''
    import os

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import matplotlib.pyplot as plt

    def plot():
        # plot base graph
        plt.plot(y_rentals)
        plt.plot(y_train, alpha=0.8, marker='o')

        # declare range
        x_range = range(train_size, train_size + test_size)

        # plot forecast graphs
        plt.plot(x_range, fcast3, label=r'$\alpha={}$ rmse={}'.format(
            fit3.model.params['smoothing_level'],
            round(rmse(fcast3, y_test), 2)))

        plt.title('{} predictions, Exponential Smoothing\n{} to {}'.format(
            test_size,
            df.index.min(),
            df.index.max()))
        plt.legend(loc='upper left')
        plt.grid()
        plt.savefig('{}/{}_exponential_smoothing.png'.format(
            draw, str(df.columns[0])))
        plt.close()

    y_rentals = df.to_numpy()
    train_size = int(len(y_rentals) * train_ratio)
    test_size = len(y_rentals) - train_size
    y_train = y_rentals[0:train_size]
    y_test = y_rentals[train_size:train_size+test_size]

    y_train = y_train.astype('double')
    fit3 = ExponentialSmoothing(y_train).fit()
    fcast3 = fit3.forecast(test_size)

    plot() if draw else None

    return y_test, fcast3


def moving_average(df, window, train_ratio=0.67, draw=False):
    '''
    args:
        df - dataframe with date as index and count as values
        draw - directory to save plot

    returns:
        draws a plot where directed
        returns lowest rmse
    '''

    import os
    import matplotlib.pyplot as plt

    def plot():
        y_real = df[df.columns[0]][-test_size:]

        plt.title('{} predictions, Simple Moving Average\n{} to {}'.format(
            test_size,
            df.index.min(),
            df.index.max()))
        plt.plot(list(range(len(df))), df[df.columns[0]], label='rentals')
        plt.plot(list(range(len(df))), rolling_mean,
                 label='{} Day SMA rmse: {}'.format(
                     window,
                     round(rmse(y_real, rolling_mean[-test_size:]), 3)),
                 color='orange')
        plt.legend(loc='upper left')
        plt.grid()
        plt.savefig('{}/{}_moving_average.png'.format(draw, str(df.columns[0])))
        plt.close()

    train_size = int(len(df) * 0.67)
    test_size = len(df) - train_size

    rolling_mean = df[df.columns[0]].rolling(window=window).mean()

    plot() if draw else None

    return df[df.columns[0]][-test_size:], rolling_mean[-test_size:]


def create_dataset(dataset, look_back=1):
    import numpy
    
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def simple_regression_univariate_lstm(dataset, optimizer, loss='mean_squared_error', layers=[],
                                      look_back=24, n_epochs=100, train_ratio=0.67):
    # simple regression
    '''
    example:
    df = pd.read_csv('darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_1/timescale_90/hex_edge_1220.63m_quantile_2_daily.csv')
    df = df.set_index('timeseries')
    df = df[df.columns[0]].to_frame()
    '''
    
    import numpy
    import time
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    
    starttime = time.time()
    method_name = 'Single Variable RNN-LSTM with Simple Regression'
    # drop the date column in order to have a df with only index and _count

    numpy.random.seed()

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into traning and test sets
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    
    if look_back > train_size or look_back > test_size:
        return None
    
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    for layer in layers:
        model.add(layer)
    model.compile(loss=loss, optimizer=optimizer)
    history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=1, verbose=1)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    totaltime = int((time.time() - starttime)/60)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    trainScore = rmse(trainY[0], trainPredict[:,0])
    print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    testScore = rmse(testY[0], testPredict[:,0])
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    
    return model, scaler.inverse_transform(dataset), trainScore, testScore, trainPredictPlot, testPredictPlot, testY, testPredict, history.history['loss'], totaltime


def univariate_lstm_with_time_regression(dataset, optimizer, loss='mean_squared_error', layers=[],
                                         look_back=24, n_epochs=100, train_ratio=0.67):
    '''
    example:
    df = pd.read_csv('darwin_rentals_time_loc_data_20180701_20190701_breakdown/quadrant_1/timescale_90/hex_edge_1220.63m_quantile_2_daily.csv')
    df = df.set_index('timeseries')
    df = df[df.columns[0]].to_frame()
    '''
    
    import numpy
    import time
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    
    numpy.random.seed()
    method = 'Single Variable RNN-LSTM with Timestep Regression Framing'
    starttime = time.time()
    # load the dataset
    dataset = dataset.values.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    
    for layer in layers:
        model.add(layer)
    
    model.compile(loss=loss, optimizer=optimizer)
    history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=1, verbose=1)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    totaltime = int((time.time() - starttime)/60)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = rmse(trainY[0], trainPredict[:,0])
    print('Train Score: %.2f RMSE' % (trainScore))

    testScore = rmse(testY[0], testPredict[:,0])
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    return model, scaler.inverse_transform(dataset), trainScore, testScore, trainPredictPlot, testPredictPlot, testY, testPredict, history.history['loss'], totaltime


def univariate_lstm_with_memory(dataset, optimizer, loss='mean_squared_error', layers=[],
                                look_back=24, n_epochs=100, train_ratio=0.67):

    import numpy
    import time
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    
    starttime = time.time()
    method = 'Stacked LSTM with memory'
    n_memory = n_epochs

    numpy.random.seed(7)
    # load the dataset
    dataset = dataset.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    
    for layer in layers:
        model.add(layer)

    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)

    history = numpy.array()

    for i in range(n_memory):
        numpy.append(history, model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False))
        model.reset_states()
    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)

    totaltime = int((time.time() - starttime)/60)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions

    return model, scaler.inverse_transform(dataset), trainScore, testScore, trainPredictPlot, testPredictPlot, testPredict, history, totaltime


def univariate_lstm_with_memory(dataset, optimizer, loss='mean_squared_error', layers=[],
                                look_back=24, n_epochs=100, train_ratio=0.67):

    import numpy
    import time
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    
    starttime = time.time()
    method = 'Stacked LSTM with memory'
    n_memory = n_epochs

    numpy.random.seed(7)
    # load the dataset
    dataset = dataset.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    # create and fit the LSTM network
    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    
    for layer in layers:
        model.add(layer)

    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)

    history = numpy.array([])

    for i in range(n_memory):
        numpy.append(history, model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False))
        model.reset_states()
    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)

    totaltime = int((time.time() - starttime)/60)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = rmse(trainY[0], trainPredict[:,0])
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = rmse(testY[0], testPredict[:,0])
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions

    return model, scaler.inverse_transform(dataset), trainScore, testScore, trainPredictPlot, testPredictPlot, testPredict, history, totaltime


def prophet_modeling(df, forecast_periods=7):
    '''
    uses fb's prophet library to forecast
    returns the model and the forecasted df
    '''

    from fbprophet import Prophet
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    return model, forecast