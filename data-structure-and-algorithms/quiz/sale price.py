import ast

suits = input()
suits = ast.literal_eval(suits)
budget = int(input())


def findSalePrice(original_price, discount_rate):
    discount_price = original_price * (discount_rate / 100)
    sale_price = original_price - discount_price
    return sale_price


def buy_expensive_suit(suits, budget):
    suits_dict = {i[1] * (1 - i[2] / 100): i[0] for i in suits}
    lst = sorted([i for i in suits_dict.keys()])
    low, high = 0, len(lst) - 1
    while True:
        guess = (low + high) // 2
        try:
            if high < low:
                return -1
            if lst[guess] < budget < lst[guess + 1]:
                return (suits_dict[lst[guess]], lst[guess])
            elif budget == lst[guess]:
                return (suits_dict[lst[guess]], lst[guess])
            elif budget < lst[guess]:
                high = guess - 1
            elif budget > lst[guess]:
                low = guess + 1
        except:
            return (suits_dict[lst[guess]], lst[guess])


suit_to_buy = buy_expensive_suit(suits, budget)
