# Regex Basics

## Character class matching:
`\d` - any digit from 0-9.  
`\D` - any character that is *not* a digit from 0-9.  
`\w` - any 'word', defined as letter, digit, or underscore.  
`\W` - any character that is *not* a letter, digit, or underscore character.  
`\s` - any space, tab, or newline character.  
`\S` - any character that is *not* a space, tab, or newline.  

## Symbols matching:
`?` - zero or one of preceding group 
`*` - zero or more of preceding group
`+` - one or more of preceding group
`{n}` - exactly n of preceding group
`{n,}` - n or more of preceding group
`{,m}` - 0 to m of preceding group
`{n,m}` - at least n and at most m of preceding group
`{n,m}?` or `*?` or `+?` - nongreedy match of preceding group
`^spam` - string must begin with spam
`spam?` - string must end with spam 
`.` - matches any character, except newline 
`[abc]` - matches any character between brackets
`[^abc]` - matches any character that isn't between brackets

## Case insensitive
`re.IGNORECASE` or `re.I` as second argument of `re.compile`

## Substitution
`namesRegex = re.compile(r'Agent \w+')`
`namesRegex.sub('CENSORED', 'Agent Alice gave the secret documents to Agent Bob.')`
Output: `'CENSORED gave the secret documents to CENSORED.'`

Using matched text as part of substitution: using `\1`, `\2` as text group in substitution:
```
agentNamesRegex = re.compile(r'Agent (\w)\w*')
agentNamesRegex.sub(r'\1****', 'Agent Alice told Agent Carol that Agent Eve knew Agent Bob was a double agent.')
```
Output: `A**** told C**** that E**** knew B**** was a double agent.'`

## Multi-Line:
```
phoneRegex = re.compile(r'''(
    (\d{3}|\(\d{3}\))?
    (\s|-|\.)?
    \d{3}
    (\s|-|\.)
    \d{4}
    (\s*(ext|x|ext.)\s*\d{2,5})?  
    )''', re.VERBOSE)
```