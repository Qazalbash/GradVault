def is_leap_year(yyyy):

    if yyyy % 4 == 0:
        if yyyy % 400 != 0 and yyyy % 100 == 0:
            return False
        return True
    return False

def next_day(yyyy,mm,dd):

    if 1<=dd<=31 and 1<=mm<=12:
      
        if mm in [1,3,5,7,8,10,12]:
          
            if dd == 31:
                dd = 1
                
                if mm == 12:  
                    yyyy += 1
                    mm = 1
                else:
                    mm += 1
            else:
                dd += 1
                
        elif mm == 2:
            if is_leap_year(yyyy):
                if dd == 29:
                    dd = 1
                    mm += 1
                else:
                    dd += 1
            elif dd == 28:
                    dd = 1
                    mm += 1
            else:
                dd += 1
        
        else:
            if dd == 30:
                dd = 1
                mm += 1
            else:
                dd += 1
        print(yyyy, mm, dd)
    
    else:
        print("invalid input")
        
yyyy, mm, dd = [int(a) for a in input().strip().split()]

next_day(yyyy, mm, dd)
