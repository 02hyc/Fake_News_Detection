import csv

# 读取第一个CSV文件中的文本和序号
text_to_num = {}
with open('dataNo.csv', encoding='utf-8') as file1:
    reader = csv.reader(file1)
    for row in reader:
        text_to_num[row[0]] = row[1]

# 读取第二个CSV文件，根据第一个文件的序号替换文本，并将结果保存到新文件
with open('dataPro.csv', encoding='utf-8') as file2, open('util-data.csv', 'w', newline='', encoding='utf-8') as output_file:
    reader = csv.reader(file2)
    writer = csv.writer(output_file)

    # 读取第一行（表头）
    header = next(reader)
    writer.writerow(header)

    # 替换文本并写入新文件
    for row in reader:
        for i in range(0, 6):
            if row[i] in text_to_num:
                row[i] = text_to_num[row[i]]
        writer.writerow(row)

