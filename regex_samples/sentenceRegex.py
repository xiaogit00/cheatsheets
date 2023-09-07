# write a regex that matches a sentence where the first word is either Alice, Bob, or Carol; 
# the second word is either eats, pets, or throws; the third word is apples, cats, 
# or baseballs; and the sentence ends with a period

import re 

sentenceRegex = re.compile(r'''(
                           (Alice|Bob|Carol)
                           \s
                           (eats|pets|throws)
                           \s
                           (apples|cats|baseballs)
                           \.
)''',re.VERBOSE)

# Tests

# Positive cases:
s1 = 'Alice eats baseballs. and '
s2 = 'Bob throws cats.'
s3 = 'Carol pets apples. And then'
# Negative cases:

s4 = 'Alice Eats Baseballs'
s5 = 'Bob throws. And'
s6 = 'Alice pets baseballs'

mo = sentenceRegex.search(s6)

print(mo.group())