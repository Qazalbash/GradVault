no_of_days = int(input())

def days_conversion(no_of_days):
  
    years = int(no_of_days/365)
    months = int((no_of_days%365)/30)
    weeks = int(((no_of_days%365)%30)/7)
    days = no_of_days - 7*weeks - 30*months - 365*years
    print("Years:",years)
    print("Months:",months)
    print("Weeks:",weeks)
    print("Days:",days)
 
days_conversion(no_of_days)
