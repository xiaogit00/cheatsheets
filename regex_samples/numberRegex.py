import re
# MY VERSION
num = '1,234'
num5 = '23,456'
num2 = '422'
num3 = '6,368,745'
num4 = '12,34,567'
numRegex = re.compile(r'''(
                    (\d{1,3}), # Case between 1 M to 10k
                    ((\d{3}),)+
                      (\d{3})
                      |
                    (\d{1,3}),
                      (\d{3}) #case between 10k to 1k
                      |
                        (\d{1,3}) # Case of less than 1000
                      )''', re.VERBOSE)

# mo = numRegex.search(num4)
# print(mo.group())

chatGPT_regex = re.compile(r'\d{1,3}(,\d{3})*$')
mo = chatGPT_regex.search(num4)
print(mo.group())