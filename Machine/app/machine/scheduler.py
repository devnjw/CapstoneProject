from .train import train
from .predict import predict_loop
from io import BytesIO
# requirement.txt 에 pip install schedule 포함
import schedule
import time
import re
import requests
import urllib
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
import datetime
import numpy as np
import pandas as pd

def Func():
    train()
    predict_loop()

def run_scheduler(TIME):
    results = {}  # {year: holidays_of_the_year_in_csv_format}

    for year in range(2021, 2022):
        res_otp = requests.get(
            'http://open.krx.co.kr/contents/COM/GenerateOTP.jspx?name=fileDown&filetype=xls&url=MKD/01/0110/01100305/mkd01100305_01&search_bas_yy=2021&gridTp=KRX&pagePath=%2Fcontents%2FMKD%2F01%2F0110%2F01100305%2FMKD01100305.jsp',
            params={
                'name': 'fileDown',
                'filetype': 'xls',
                'url': 'MKD/01/0110/01100305/mkd01100305_01',
                'search_bas_yy': year,
            },
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'}
        )
        otp = res_otp.text
        res_byte = requests.post(
            'http://file.krx.co.kr/download.jspx',
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Referer': 'http://open.krx.co.kr/',
            },
            data=urllib.parse.urlencode({'code':otp})
        )
        df = pd.read_excel(BytesIO(res_byte.content))
        df = df.to_csv(index=False, header=False)

        results[year] = df

    holidays = []
    for year in results:
        holidays.extend(
            re.findall(
                r'(\d+)-(\d+)-(\d+)(?:\s\(.*?\))?\,(.*?)\s*\,(.*)',
                results[year]
            )
        )

    class KRTradingCalendar(AbstractHolidayCalendar):
        rules = [
            Holiday(holiday[2], year=int(holiday[0]), month=int(holiday[1]), day=int(holiday[2]))
            for holiday in holidays
        ]

    TDay = TradingDay = CustomBusinessDay(calendar=KRTradingCalendar())

    date_today = datetime.date.today()
    date_yesterday = date_today - datetime.timedelta(days=1)

    if date_today == date_yesterday + TDay :

        schedule.every().day.at(TIME).do(Func)

        while True:
            schedule.run_pending()
            time.sleep(1)
