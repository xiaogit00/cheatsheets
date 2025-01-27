# Regex Basics

`import re`
`re.compile(r'\w+').findall('a text')`

## Character class matching:
`\d` - any digit from 0-9.  
`\D` - any character that is *not* a digit from 0-9.  
`\w` - any 'word', defined as letter, digit, or underscore.  
`\W` - any character that is *not* a letter, digit, or underscore character.  
`\s` - any space, tab, or newline character.  
`\S` - any character that is *not* a space, tab, or newline. 
`\b` - word boundary - '\b\w\w\w\b' - will match 3 character words; if not, will match anything with 3 letters 

### More on word boundaries
- matches sth in between a word char \w and non-word char. 

## Symbols matching:
`?` - 0 or one of preceding group  
`*` - 0 or more of preceding group  
`+` - 1 or more of preceding group  
`{n}` - exactly n of preceding group  
`{n,}` - n or more of preceding group  
`{,m}` - 0 to m of preceding group  
`{n,m}` - at least n and at most m of preceding group  
`{n,m}?` or `*?` or `+?` - nongreedy match of preceding group  
`^spam` - string must begin with spam  
`spam?` - string must end with spam  
`.` - matches any character, except newline  
`[,.;:]` - matches any character between brackets  (character sets)
`[^abc]` - matches any character that's opps of those here

## Findall
`phoneRegex.findall(text)` - returns an array of matches  

## Case insensitive
`re.IGNORECASE` or `re.I` as second argument of `re.compile`

## Substitution
```
namesRegex = re.compile(r'Agent \w+')
namesRegex.sub('CENSORED', 'Agent Alice gave the secret documents to Agent Bob.')
```

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

## Using Regex in Python
```
import re

phoneRegex = re.compile(r'\d\d\d\d\d')

mo = phoneRegex.search(text)

print(mo.group())

```
## Groups
You can match something, but then capture groups. Using ( )
- groups can be nested according to number, or opening bracket (1 (2 ) (3 (4)) ) 
- `([\w.-]+)@([\w.-])` - email capture e.g. alice@gmail.com
    - match.group(1); 

Naming groups:
`(?P<user>[\w.-]+)@(?P<domain>[\w.-])` - match.group('user')

## Backreference \1
Reference back to the first 

## Lookarounds:
(?=): positive lookahead -- A(?=B) finds expression A but only when followed by expression B. E.g. `\w+(?= Simpson)` 

(?!): negative lookahead -- A(?!B) finds expression A but only when not followed by expression B. 
- `\w+(?! Simpson)` 
- `\b(?!2)\d{3}\b` -> 3 digit characters that don't begin with 2. 
- `(?!SBS|SH|SMC|SEP|SPF|STC)[A-Z]{3}\d{4}[A-Z]` -> 3 letters that don't begin with the above

(?<=): positive lookbehind -- (?<=B)A finds expression A but only when preceded by expression B  `(?<=S\$)[0-9,.]*\d` -> 

(?<!): negative lookbehind -- (?<!B)A finds expression A but only when not preceded by expression B

## Lookaheads (?=) -> read: behind 
Let's say you have this sentence: 
> "The team consists of Homer Simpson, Barney Gumble, Monty Burns, Marge Simpson, Ned Flanders, and Lenny Lennard."

You want to match all kinds of Simpsons. How would you do it with traditional regex? You can do it with:

`re.compile(r'(\w+) Simpson').findall(t1)`

With lookaheads, it looks like this:
`re.compile(r'\w+(?= Simpson)').findall(t1)`

Both will return `['Homer', 'Marge']`, but there's a key difference. 

For lookaheads, "Simpson" is *not* included in the match. 

Whereas capturing groups (first one) will require Simpson to be part of the match. 

A practical difference lies in overlapping matches. Lookaheads will not *consume* the word you want to match, so it can be 'rematched'. Whereas capturing groups only capture a match once. 

## Lookbehinds (?<=) -read: infront equal
Let's say you want to match all S$ values (e.g. S$100,000), but return only the number. 

Without lookbehinds, you can't do it. This regex will match everything:
`S\$[0-9,.]*\d`

i.e. Input: For 5 years, the funding is $1,000,000.00 which converts to S$1,307,600.00.
Output: S$1,307,600.00

If you use lookbehinds, `(?<=S\$)[0-9,.]*\d`, it'll output: `1,307,600.00`

## Common hacks
`[0-9][0-9]` - 2 digits, e.g. **14** 
`[a-zA-Z0-9]+` - letters and digits of length 1 or more (\w contains underscore)
`[Mm]onth[s]?` - match Month, month, months, Months
`lo+l` - e.g. loool, looooool
`([\w.-]+)@([\w.-])` - email capture e.g. alice@gmail.com
`(ha){2,}` - finds all kinds of haha, hahaha hahahahah
`(http[s]://)([\w.-]*)(/[\w\-]+)` - domain, news.nus.edu.sg  
`'$[0-9,.]*\d'` - matches numbers like 100,323,132.00- ensures that this ends with a number

## Philosophy of regex
- finding 'perfect' regex can be difficult 
    -  'perfect' = matches all corner cases
- fuzzy matches can be better

