import numpy as np
import string
import re

"""
file: reddit_tags.py
---
Detects Reddit-Specific HTML tags and other features
"""

"""
function: count_all_caps

Returns the number of all-caps words in a message
"""

def count_all_caps(text):
    words = text.split()
    # Check if the word is all uppercase, is alphabetical, and has more than one letter. Differentiating all caps vs acronyms?
    all_caps_count = sum(
        1 for word in words 
        if word.strip(string.punctuation).isupper()  
        and word.strip(string.punctuation).isalpha() 
        and len(word.strip(string.punctuation)) > 1 
        or (word.strip(string.punctuation) != "I" and len(word.strip(string.punctuation)) == 1)
    ) 
    return all_caps_count

"""
function: count_links

Returns the number of links in a message.
"""
def count_links(text):
    plain_urls = re.findall(r'http[s]?://[^\s]+', text)
    return len(plain_urls)

"""
function: count_user_references

Returns the number of user references in a message, indicated by format u/username.
"""
def count_user_references(text):
    user_refs = re.findall(r'\bu/[^\s]+', text)
    return len(user_refs)

"""
function: count_bold

Returns the number of **bolded**, *italicized*, or ***bolded and italicized*** words in a message.
"""
def count_emphasis(text):
    formatted_texts = re.findall(r'\*{1,3}(.*?)\*{1,3}', text)
    return len(formatted_texts)

"""
function: count_bullet_points

Returns the number of bullet points in a message starting with asterisks or dashes.
"""
def count_bullet_points(text):
    bullet_points = re.findall(r'^[\*\-] .+', text, flags=re.MULTILINE)
    return len(bullet_points)

"""
function: count_numbering

Returns the number of numberings in a message, indicated by a format like "1. ".
"""
def count_numbering(text):
    numbering = re.findall(r'^\d+\. .+', text, flags=re.MULTILINE)
    return len(numbering)

"""
function: count_line_breaks

Returns the number of paragraphs / line breaks in a message.
"""
def count_line_breaks(text):
    normalized_text = re.sub(r'\r\n?', '\n', text)
    text_single_breaks = re.sub(r'\n+', '\n', normalized_text)
    return text_single_breaks.count('\n') + 1

"""
function: count_quotes

Returns the number instances of text enclosed in quotation marks in a message.
"""
def count_quotes(text):
    quotes = re.findall(r'"([^"]*)"|\'(\b[^\'\s]{3,}[^\'\s]*\b|[^\']+\b[^\'\s]{3,}\b)\'', text)
    return len(quotes)

"""
function: is_responding_to_someone

Returns a boolean indicating if the message is quoting someone else, as indicated by ">" or "&gt;".
"""
def count_responding_to_someone(text):
    return len(re.findall(r'^(>|\&gt;)', text))

"""
function: count_ellipses

Returns the number of ellipses (three or more consecutive dots) in a message.
"""
def count_ellipses(text):
    ellipses = re.findall(r'\.{3,}', text)
    return len(ellipses)

"""
function: count_parentheses

Returns the number of instances of text enclosed in parentheses in a message.
"""
def count_parentheses(text):
    text_in_parentheses = re.findall(r'\(([^)]*)\)', text)
    return len(text_in_parentheses)

"""
function: count_emojis

Returns the number of instances of emojis in a message.
"""
def count_emojis(text):
    emoji_pattern = r'[:;]-?\)+'
    emojis = re.findall(emoji_pattern, text)
    return len(emojis)


# print(count_all_caps("HELLO WORLD, THIS IS A TEST. hi HI. hi HI hi HI"))  # Test count_all_caps

# print(count_links("Check out this [link](https://example.com) and this one http://example.org"))  # Test count_links

# print(count_user_references("Hello u/user1 and u/user2, hi hi hi?"))  # Test count_user_references

# print(count_emphasis("This is **bold**, *italics*, and this is not. This is ***bolded and italicized***"))  # Test count_emphasis

# print(count_bullet_points("* item 1\n* item 2\n- item 3	"))  # Test count_bullet_points

# print(count_numbering("1. First\n2. Second\n3. Third"))  # Test count_numbering

print(count_line_breaks("This is the first line.\nThis is the second line.\nThis is the third line."))  # Test count_line_breaks

# print(count_line_breaks("I have a line\n\n\n\n\nhere is a new line"))  # Test count_line_breaks

print(count_quotes("\"This is a quote.\" She said, \"Here's another.\""))  # Test count_quotes

print(count_quotes("\"I can't believe you use single quotes to quote people,\" she said. \"Well, he replied, \'sometimes single quotes are useful when you nest quotes inside other quotes,\' according to my English teacher\" Then she said: \'okay\'"))  # Test count_quotes

# print(is_responding_to_someone("> Quoting someone else\nThis is my reply."))  # Test is_responding_to_someone

# print(is_responding_to_someone("&gt; Quoting someone else\nThis is my reply."))  # Test is_responding_to_someone

# print(count_ellipses("Well... I'm not sure... Maybe..."))  # Test count_ellipses

# print(count_parentheses("This is a sentence (with some text in parentheses)."))  # Test count_parentheses
