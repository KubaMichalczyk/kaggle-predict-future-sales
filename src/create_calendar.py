import pandas as pd
import datetime as dt
import holidays

start_date = dt.datetime.strptime("2013-01-01", "%Y-%m-%d")
end_date = dt.datetime.strptime("2015-11-30", "%Y-%m-%d")

dates = [start_date + dt.timedelta(days=x) for x in range(0, (end_date - start_date + dt.timedelta(days=1)).days)]

ru_holidays = holidays.Russia()

calendar = pd.Series(dates).rename("date").to_frame()

calendar["bank_holiday"] = calendar["date"].apply(lambda x: ru_holidays.get(x))
calendar["weekday"] = calendar["date"].apply(lambda x: dt.date.isoweekday(x))

# In Russia, if the date of bank holiday observance falls on a weekend, the following Monday will be a day off in lieu
# of the holiday. I think the exception is New Year Holiday as it lasts from 1st to 8th January and additional day is
# not given.

days_in_lieu = calendar.loc[calendar["bank_holiday"].notnull() &
                            calendar["weekday"].isin([6, 7]) &
                            (calendar["bank_holiday"] != "Новый год")].copy()
days_in_lieu["date"] = days_in_lieu.apply(lambda x: x["date"] + dt.timedelta(days=7 - x["weekday"] + 1), axis=1)

calendar = calendar.merge(days_in_lieu, how="left", on="date", suffixes=["", "_in_lieu"])
calendar["bank_holiday"] = calendar["bank_holiday"].fillna(calendar["bank_holiday_in_lieu"])
calendar.drop(["bank_holiday_in_lieu", "weekday_in_lieu"], axis=1, inplace=True)

calendar.to_csv("../auxiliaries/bank_holidays_calendar.csv", index=False)
