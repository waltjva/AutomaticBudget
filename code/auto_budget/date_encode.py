import datetime as dt
import numpy as np

total_day_seconds = 24*60*60

class date_encode:
    # def __init__(self, date):
    #     self.date = date

    def date_each(self, date):
        """encode each date point and incorporate circular nature of days/months"""
        year = date.year
        sin_month = np.sin(2 * np.pi * date.month / 12)
        cos_month = np.cos(2 * np.pi * date.month / 12)
        sin_day = np.sin(2 * np.pi * date.month / 31)
        cos_day = np.cos(2 * np.pi * date.month / 31)
        return year, sin_month, cos_month, sin_day, cos_day

    def yield_seconds(self, x):
        return dt.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()

    # df['PD_year'] = df['Purchase Date'].apply(lambda x: x.year)
    # df['sin_PD_month'] = np.sin(2 * np.pi * df['Purchase Date'].apply(lambda x: x.month) / 12)
    # df['cos_PD_month'] = np.cos(2 * np.pi * df['Purchase Date'].apply(lambda x: x.month) / 12)
    # df['sin_PD_day'] = np.sin(2 * np.pi * df['Purchase Date'].apply(lambda x: x.day) / 31)
    # df['cos_PD_day'] = np.cos(2 * np.pi * df['Purchase Date'].apply(lambda x: x.day) / 31)
    # df['PT_total_seconds'] = df['Purchase Time'].apply(de.yield_seconds)
    # df['sin_PT_total_seconds'] = np.sin(2 * np.pi * df['PT_total_seconds'] / total_day_seconds)
    # df['cos_PT_total_seconds'] = np.cos(2 * np.pi * df['PT_total_seconds'] / total_day_seconds)
    # df['VD_year'] = df['Verification Date'].apply(lambda x: x.year)
    # df['sin_VD_month'] = np.sin(2 * np.pi * df['Verification Date'].apply(lambda x: x.month) / 12)
    # df['cos_VD_month'] = np.cos(2 * np.pi * df['Verification Date'].apply(lambda x: x.month) / 12)
    # df['sin_VD_day'] = np.sin(2 * np.pi * df['Verification Date'].apply(lambda x: x.day) / 31)
    # df['cos_VD_day'] = np.cos(2 * np.pi * df['Verification Date'].apply(lambda x: x.day) / 31)