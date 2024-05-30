import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


class CryptoPredictor:
    def __init__(self, file_path, currency):
        self.file_path = file_path
        self.currency = currency
        self.data = None
        self.model = None
        self.train = None
        self.test = None
        self.forecast = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data = self.data[self.data['Currency'] == self.currency]
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.data = self.data.asfreq('D')  # Установка частоты данных на дневную

    def prepare_data(self):
        close_prices = self.data['Close']
        close_prices_diff = close_prices.diff().dropna()  # Применение дифференцирования
        train_size = int(len(close_prices_diff) * 0.8)
        self.train, self.test = close_prices_diff[:train_size], close_prices_diff[train_size:]
        self.train_original = self.data['Close'][:train_size]  # Оригинальные цены для восстановления

    def build_model(self, order=(5, 1, 0)):
        self.model = ARIMA(self.train, order=order)
        self.model_fit = self.model.fit()

    def make_forecast(self):
        self.forecast_diff = self.model_fit.forecast(steps=len(self.test))
        self.forecast = self.forecast_diff.cumsum() + self.train_original.iloc[-1]  # Обратное дифференцирование

    def plot_train_data(self, save_path='train_data.png'):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_original.index, self.train_original, label='Train')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.currency} - Training Data')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_test_data_forecast(self, save_path='test_data_forecast.png'):
        plt.figure(figsize=(12, 6))
        plt.plot(self.test.index, self.data['Close'][self.test.index], label='Test')
        plt.plot(self.test.index, self.forecast, label='Forecast', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.currency} - Test Data & Forecast')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def test_stationarity(timeseries):
    # Выполнение теста Дики-Фуллера
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


# Использование класса
predictor = CryptoPredictor('ETH-BTC-USD.csv', 'Bitcoin')
predictor.load_data()
predictor.prepare_data()
test_stationarity(predictor.train)

# Построение модели и прогноз
predictor.build_model(order=(5, 1, 1))
predictor.make_forecast()

# Создание и сохранение графиков
predictor.plot_train_data('train_data.png')
predictor.plot_test_data_forecast('test_data_forecast.png')
