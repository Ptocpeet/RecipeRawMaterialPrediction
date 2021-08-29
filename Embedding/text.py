import csv

f = open('result.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)


def write(path):
    with open(path, encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            csv_writer.writerow(row)


write('../result/result1.csv')
write('../result/result2.csv')
write('../result/result3.csv')
write('../result/result4.csv')
f.close()
