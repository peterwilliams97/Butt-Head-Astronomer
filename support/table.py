from pprint import pprint

text = '''
    0: ASC
    1: Manufacturer
    2: Super Region
    3: Parent Region
    4: Reseller Region
    5: Reseller Tier
    6: Reseller Code
    7: Reseller Name
    8: Reseller City
    9: Reseller State
    10: Reseller Country
    11: Reseller Enabled/Disabled
    12: Customer Organisation
    13: Customer City
    14: Customer Country
    15: id
    16: address1
    17: address2
    18: city
    19: country
    20: state
    21: zip
    22: firstName
    23: lastName
    24: organizationName
    25: clientPurchaseOrderNumber
    26: currency
    27: dateCreated
    28: electedPaymentMethod
    29: isQuote
    30: orderNumber
    31: orderProductLine
    32: orderStatus
    33: orderSystemEntryMethod
    34: orderSystemIpAddress
    35: orgType
    36: discountPercentage
    37: discountReason
    38: discountType
    39: paymentSecurityId
    40: paymentStatus
    41: priceAtOrderTime
    42: quoteNumber
    43: resellerCode
    44: resellerDiscountPercentage
    45: specialOrderNumber
    46: subreseller
    47: subSubReseller
    48: type
    49: customerId
    50: customerComment
    51: exchangeRateAtOrderTime
    52: resellerDiscountReason
    53: emailAddress
    54: exportedToNetSuite
    55: endCustomerId
    56: originalEndCustomerId
'''

lines = text.split('\n')
parts = [ln.split(':') for ln in lines]
lines = [p[1] for p in parts if len(p) == 2]
lines = [ln.strip() for ln in lines]
lines = [ln for ln in lines if ln]
print('%d lines' % len(lines))
# d = {k: str for k in lines}
# pprint(d)
print('{\n%s\n}' % '\n'.join("    '%s': str," % ln for ln in lines))
