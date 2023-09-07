import pyperclip , re

# text = pyperclip.paste()

text = '81293902 is my number, and my email is lei@gmail.com, sometimes 99991132 and good@hotmail.com'

phoneRegex = re.compile(r'\d{8}')
emailRegex = re.compile(r'\w+@\w+.com')

mo = phoneRegex.findall(text)
mo2 = emailRegex.findall(text)

numbersText = 'Numbers:\n'
emailsText = 'Emails:\n'

if mo:
    for num in mo:
        numbersText += num + '\n'
else:
    numbersText += "No numbers found"

if mo2:
    for email in mo2:
        emailsText += email + '\n'
else:
    emailsText += "No emails found"

found = numbersText + '\n' + emailsText
pyperclip.copy(found)
