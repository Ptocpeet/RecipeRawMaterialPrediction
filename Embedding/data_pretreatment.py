import csv
import re
import numpy as np
from pattern3.text.en import singularize

materials = []
NotIn = []  # 测试集中训练集中没有的
material_index = {}  # 食材的标号 例如:{eggs:0,...}
material_sum = {}  # 对应食材标号的食材质量总和
material_index_sum = {}  # 单类食材存在个数,用于后续数据填充
material_index_probability = {}  # 单类食材的出现频率,单类食材出现次数/总次数
material_num_list = {}  # 单类食材数量列表，用来清理数据
material_data_border = {}  # 单类食材的u+3σ值，用于数据清理
recipe_normalize_data = []  # 菜谱归一化后数据
recipe_embedding_vec = []  # 菜谱one-hot编码后向量


def data_read(path):  # 从文件中读取数据
    train_data = []
    with open(path, encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            train_data.append(row)
    return train_data


def data_split(datas):  # 将读取的数据处理为字典列表
    result = []
    for data in datas:
        dst = {}
        for index in range(2, len(data)):
            dec = data[index].split('#')
            temp = dec[0]
            temp = re.sub('fresh |frozen |large |small |chunks ', '', temp)  # 去掉部分无关紧要形容词
            temp = singularize(temp)
            if temp not in materials:
                materials.append(temp)
            dst[temp] = dec[1]
        if len(dst) == 0:  # 去除缺失值
            continue
        result.append(dst)
    return result


def data_split2(datas):  # 将读取的数据处理为字典列表
    result = []
    for data in datas:
        dst = []
        temp_ar = [data[0], data[1]]
        for index in range(2, len(data)):
            dec = data[index].split('#')
            dst.append(dec[0])
        temp_ar.append(dst)
        result.append(temp_ar)
    return result


def text():
    with open('../recipe3.csv', encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            x = 0
            for item in row:
                x += 1
                if x <= 2:
                    continue
                dec = item.split('#')
                if dec[0] not in materials:
                    NotIn.append(dec[0])


def get_material_information(data):  # 获取食材信息
    index = 0
    material_total = 0
    for recipe in data:  # 数据预处理
        for material in recipe:
            temp = 0
            sav = recipe[material]
            recipe[material] = recipe[material].lower()
            if recipe[material]:
                if 'tbsp' in recipe[material]:
                    if 'honey' in material:
                        temp = float(re.sub('tbsp', '', recipe[material])) * 30
                    else:
                        temp = float(re.sub('tbsp', '', recipe[material])) * 15
                elif 'tsp' in recipe[material]:
                    temp = float(re.sub('tsp', '', recipe[material])) * 6
                elif 'tbs' in recipe[material]:
                    temp = float(re.sub('tbs', '', recipe[material])) * 15
                elif 'tablespoons' in recipe[material]:
                    temp = float(re.sub('tablespoons', '', recipe[material])) * 15
                elif 'tablespoon' in recipe[material]:
                    temp = float(re.sub('tablespoon', '', recipe[material])) * 15
                elif 'teaspoon' in recipe[material]:
                    temp = float(re.sub('teaspoon', '', recipe[material])) * 5
                elif 'dozen' in recipe[material]:
                    temp = float(re.sub('dozen', '', recipe[material])) * 12
                elif 'oz' in recipe[material]:
                    temp = float(re.sub('oz', '', recipe[material])) * 29.57
                elif 'can' in recipe[material]:
                    temp = float(re.sub('can', '', recipe[material])) * 446.25
                elif 'lbs' in recipe[material]:
                    temp = float(re.sub('lbs', '', recipe[material])) * 453.59
                elif 'lb' in recipe[material]:
                    temp = float(re.sub('lb', '', recipe[material])) * 453.59
                elif 'cups' in recipe[material]:
                    temp = float(re.sub('cups', '', recipe[material])) * 180
                elif 'cup' in recipe[material]:
                    temp = float(re.sub('cup', '', recipe[material])) * 180
                elif 'ml' in recipe[material]:
                    temp = float(re.sub('ml', '', recipe[material]))
                elif 'tbls' in recipe[material]:
                    temp = float(re.sub('tbls', '', recipe[material])) * 15
                elif 'l' in recipe[material]:
                    temp = float(re.sub('l', '', recipe[material])) * 1000
                elif 'jgger' in recipe[material]:
                    temp = float(re.sub('jgger', '', recipe[material])) * 45
                elif 'jigger' in recipe[material]:
                    temp = float(re.sub('jigger', '', recipe[material])) * 45
                elif 'g' in recipe[material]:
                    temp = float(re.sub('g', '', recipe[material]))
                elif re.search('c$', recipe[material]):
                    temp = float(re.sub('c', '', recipe[material])) * 210
                elif re.search('t$', recipe[material]):
                    if re.search('T$', sav):
                        temp = float(re.sub('T', '', sav)) * 15
                    else:
                        recipe[material] = ''
                        continue
                elif '¾' in recipe[material]:
                    temp = 0
                elif '½' in recipe[material]:
                    temp = 0
                elif '1–1' in recipe[material]:
                    temp = 0
                elif '¼' in recipe[material]:
                    temp = 0
                elif 'to6' in recipe[material]:
                    temp = 0
                elif '0ne' in recipe[material]:
                    temp = 0
                else:
                    temp = float(recipe[material])
            if recipe[material]:
                recipe[material] = temp
            if material not in material_index_sum:
                if temp:
                    material_num_list[material] = [temp]
                    material_index_sum[material] = 1
            else:
                if temp:
                    material_num_list[material].append(temp)
                    material_index_sum[material] += 1

            if material not in material_index:
                material_index[material] = index
                index += 1
                material_sum[material] = temp
            else:
                material_sum[material] += temp

            material_total += 1

    for i in material_index_sum:  # 获得单类食材的出现词频
        material_index_probability[i] = material_index_sum[i] / material_total
    print("向量维数:", index)
    return index


def clean_data(data):
    for material in material_index:
        if material not in material_num_list:
            continue
        num_mean = np.mean(material_num_list[material])
        num_std = np.std(material_num_list[material])
        material_data_border[material] = [num_mean - 3 * num_std, num_mean + 6 * num_std]
    for index, recipe in enumerate(data):  # 数据清理
        total_num = 0
        for material in recipe:
            if not recipe[material]:
                continue
            if material not in material_num_list:
                continue
            if recipe[material] > material_data_border[material][1]:
                total_num += 1
        if total_num >= 2:
            topx = 10
            temp_total = total_num
            while temp_total > 1:
                temp_total = 0
                for material in recipe:
                    if not recipe[material]:
                        continue
                    if material not in material_num_list:
                        continue
                    material_sum[material] -= recipe[material] * (1 - 1 / topx)
                    recipe[material] = recipe[material] / topx
                    if recipe[material] > material_data_border[material][1] * 3 / 10:
                        temp_total += 1
                topx /= 2
            for material in recipe:
                if not recipe[material]:
                    continue
                if material not in material_num_list:
                    continue
                if recipe[material] > material_data_border[material][1] * 3 / 10:
                    material_sum[material] -= recipe[material]
                    material_index_sum[material] -= 1
                    recipe[material] = ''
        else:
            for material in recipe:
                if not recipe[material]:
                    continue
                if material not in material_num_list:
                    continue
                if recipe[material] > material_data_border[material][1]:
                    material_sum[material] -= recipe[material] * 9 / 10
                    recipe[material] = recipe[material] / 10
                    if recipe[material] > material_data_border[material][1] * 3 / 10:
                        material_sum[material] -= recipe[material]
                        material_index_sum[material] -= 1
                        recipe[material] = ''


def normalize(data):  # 归一化
    for recipe in data:
        for material in recipe:
            if recipe[material]:
                recipe[material] = recipe[material] / material_sum[material]


def get_recipe_embedding(total, data):  # 获取菜谱one-hot编码
    for recipe in data:
        embedding_vec = np.zeros(total, np.int8)
        for material in recipe:
            embedding_vec[material_index[material]] = 1
        recipe_embedding_vec.append(embedding_vec)
