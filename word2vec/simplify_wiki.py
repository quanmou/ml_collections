import re
import codecs
import tools.text_utils as tools


# 可以加入到tokenize_wiki.py里面
def filter(input_file):
    outfile = codecs.open('./wiki.jian.txt', 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as myfile:
        count = 1
        for line in myfile:
            if re.match('\s*\n$', line):
                continue
            line = tools.traditional2simplified(line)
            outfile.write(line)
            print(str(count) + ': ' + line)
            count += 1
    outfile.close()


if __name__ == '__main__':
    filter('./wiki.txt')
