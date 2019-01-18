import re
import codecs
import tools.text_utils as tools


def filter(input_file):
    # p1 = re.compile('（）')
    # p2 = re.compile('《》')
    p3 = re.compile('「')
    p4 = re.compile('」')
    p5 = re.compile('<doc (.*)>')
    p6 = re.compile('</doc>')
    outfile = codecs.open('../data/cn/wiki_00_jian.txt', 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as myfile:
        count = 1
        for line in myfile:
            if re.match('\s*\n$', line):
                continue
            # line = p1.sub('', line)
            # line = p2.sub('', line)
            line = p3.sub('', line)
            line = p4.sub('', line)
            line = p5.sub('', line)
            line = p6.sub('', line)
            line = tools.traditional2simplified(line)
            outfile.write(line)
            print(str(count) + ': ' + line)
            count += 1
    outfile.close()


if __name__ == '__main__':
    # input_file = sys.argv[1]
    filter('../data/cn/extracted/AA/wiki_00')
