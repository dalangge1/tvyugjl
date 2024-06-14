import re

from LAC import LAC

# 装载LAC模型
lac = LAC(mode='lac')


def remove_brackets_and_contents(text):
       return re.sub(r"\[.*?\]", "", text)

def participe_def(text):
       text = remove_brackets_and_contents(text)
       lac_result = lac.run(text.replace(" ", ""))
       print(lac_result)
       tag_to_keep = ['PER', 'LOC', 'ORG', 'TIME', 'a', 'ad', 'an', 'n', 'f', 's', 'nw', 'm', 'nz', 'vd', 'v', 'vn']
       filtered_words = []
       for word, tag in zip(lac_result[0], lac_result[1]):
              if tag in tag_to_keep:
                     if word not in filtered_words:
                            filtered_words.append(word)

       result_string = ', '.join(filtered_words)
       return result_string

def participe_perpro_name(text):
       lac_result = lac.run(text.replace(" ", ""))
       print(lac_result)
       tag_to_keep = ['PER']
       filtered_words = []
       for word, tag in zip(lac_result[0], lac_result[1]):
              if tag in tag_to_keep:
                     if word not in filtered_words:
                            filtered_words.append(word)

       #result_string = ', '.join(filtered_words)
       return filtered_words