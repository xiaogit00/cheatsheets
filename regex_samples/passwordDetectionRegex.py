# Write a function that uses regular expressions to make sure the password string it is passed is strong. A strong password is defined as one that is at least eight characters long, contains both uppercase and lowercase characters, and has at least one digit. You may need to test the string against multiple regex patterns to validate its strength.

import re

charRegex = re.compile(r'.{8,}')
lowerRegex = re.compile(r'[a-z]')
upperRegex = re.compile(r'[A-Z]')
digitRegex = re.compile(r'\d')

def checkPassword(p):
    mo0 = charRegex.search(p)
    mo = lowerRegex.search(p)
    mo2 = upperRegex.search(p)
    mo3 = digitRegex.search(p)

    if mo0 == None:
        print("Password is less than 8 digits long.")
        return
    if mo == None:
        print("Password requires a lower case.")
        return 
    if mo2 == None:
        print("Password requires an upper case.")
        return
    if mo3 == None:
        print("Password requires a digit.")
        return
    print("Password is valid")
    return True

checkPassword("iI93jjii")
