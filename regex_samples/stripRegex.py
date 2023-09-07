# Write a function that takes a string and does the same thing as the strip() string method. If no other arguments are passed other than the string to strip, then whitespace characters will be removed from the beginning and end of the string. Otherwise, the characters specified in the second argument to the function will be removed from the string.
import re


def strip(str, s=None):
    if s:
        removeCharRegex = re.compile(s)
        output = removeCharRegex.sub('', str)
        print(output)
    else:
        cleanWordRegex = re.compile(r'\s*(.*)\s*')
        mo = cleanWordRegex.search(str)
        print(mo.group(1))

strip("I love this cake", "e")