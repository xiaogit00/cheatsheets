import re

# First name capitalised, then, Nakamoto

# Positive cases
name1 = 'Satoshi Nakamoto'
name2 = 'Alice Nakamoto'
name3 = 'RoboCop Nakamoto'

# Negative cases
name4 = 'satoshi Nakamoto'
name5 = 'Mr. Nakamoto'
name6 = 'Nakamoto'
name7 = 'Satoshi nakamoto'

fullNameRegex = re.compile(r'[A-Z]\w+\sNakamoto')

mo = fullNameRegex.search(name3)

print(mo.group())