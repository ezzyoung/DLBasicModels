from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

def parser(x): #시간 표현하는 함수 정의
    return datetime.strptime('199'+x, '%Y-&m') #날짜와 시간 정보를 문자열로 바꾸어 주는 메서드

series = read_csv('C:\Users\ilsai\MLModels\Time-Series\chap07-data\sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = DataFrame(model_fit.resid)
residuals=DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())