import data_pretreatment as dp
import countPSME as cP
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import numpy as np
from pattern3.text.en import singularize
import re
from numba import jit
import csv

f = open('result.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)

model = ""
material_vec_size = 300
material_index_probability = {}
material_index = {}
material_sum = {}  # 对应食材标号的食材质量总和
material_index_sum = {}  # 单类食材存在个数,用于后续数据填充


def get_pw(material):
    if material in material_index_probability:
        return material_index_probability[material]
    else:
        return 1.0


def recipe2vec(data):
    a = 1e-3
    recipe_set = []
    for recipe in data:
        vs = np.zeros(material_vec_size)
        recipe_size = len(recipe)
        for material in recipe:
            factor = a / (a + get_pw(material))
            if material not in material_index:
                continue
            vs = np.add(vs, np.multiply(factor, model[material]))
        vs = np.divide(vs, recipe_size)
        recipe_set.append(vs)

    # 计算主成分分析
    # iu = 0  # 检查缺失值
    # for i in recipe_set:
    #     if np.isnan(i).any():
    #         print(i)
    #         print(iu)
    #     iu += 1
    pca = PCA(n_components=material_vec_size)
    pca.fit(np.array(recipe_set))
    u = pca.components_[0]
    uut = np.multiply(u, np.transpose(u))

    # 计算最终食谱向量
    recipe_vecs = []
    for vs in recipe_set:
        recipe_vecs.append(np.subtract(vs, np.multiply(uut, vs)))

    return recipe_vecs


def cos_sim(vector_a, vector_b):  # 计算余弦相似度
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim


def check(temp):
    temp = re.sub('fresh |frozen |large |small |chunks ', '', temp)  # 去掉部分无关紧要形容词
    temp = singularize(temp)
    return temp


def isFloat(x):
    try:
        float(x)
        return True
    except:
        return False


def get_ans(vec_data1, vec_data2, data1, data2):
    count = 0
    place = 0

    # total_sum = 0
    # n_ = 0
    # print(n_)

    for index1, recipe2 in enumerate(vec_data2):
        top10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 记录相似度前10的菜谱余弦值
        recipe_top10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        while len(text_data_origin[place][2]) == 0:
            write_arr = [text_data_origin[place][0], text_data_origin[place][1]]
            csv_writer.writerow(write_arr)
            place += 1
        for index2, recipe1 in enumerate(vec_data1):
            cos = cos_sim(recipe1, recipe2)
            for i, item in enumerate(top10):
                if cos > item:
                    for j in range(10, -1, i + 1):
                        top10[j] = top10[j - 1]
                        recipe_top10[j] = recipe_top10[j - 1]
                    top10[i] = cos
                    recipe_top10[i] = index2
                    break

        temp_or_dit = {}
        for material in data2[index1]:
            if material not in material_index_sum:
                continue

            # if not data2[index1][material]:
            #     continue
            # if not isFloat(data2[index1][material]):
            #     continue
            # source_value = float(data2[index1][material])

            material_similar = model.wv.most_similar(material)
            temp_sum_value = 0
            temp_sum_index = 0
            for mt_index, i in enumerate(recipe_top10):
                if top10[mt_index] < 0.75:
                    break
                source_recipe = data1[i]
                if material in source_recipe:
                    if source_recipe[material]:
                        temp_sum_index += 1
                        temp_sum_value += source_recipe[material]
                    else:
                        for data1_material in source_recipe:
                            if source_recipe[data1_material]:
                                # temp_index1 = material_index[material]
                                # temp_index2 = material_index[data1_material]
                                temp_sum_index += 1
                                x1_ = material_sum[material] / material_index_sum[material]
                                x2_ = material_sum[data1_material] / material_index_sum[data1_material]
                                temp_sum_value += x1_ * source_recipe[data1_material] / x2_
                                break
                    continue
                flag = 0
                for en_item in material_similar:
                    en_material = en_item[0]
                    if en_material in source_recipe:
                        if source_recipe[en_material]:
                            # temp_index1 = material_index[material]
                            # temp_index2 = material_index[en_material]
                            temp_sum_index += 1
                            x1_ = material_sum[material] / material_index_sum[material]
                            x2_ = material_sum[en_material] / material_index_sum[en_material]
                            temp_sum_value += x1_ * source_recipe[en_material] / x2_
                            flag += 1
                            break
                if flag:
                    continue
                else:
                    for data1_material in source_recipe:
                        if source_recipe[data1_material]:
                            temp_sum_index += 1
                            x1_ = material_sum[material] / material_index_sum[material]
                            x2_ = material_sum[data1_material] / material_index_sum[data1_material]
                            temp_sum_value += x1_ * source_recipe[data1_material] / x2_
                            break
            if temp_sum_index == 0:
                x1 = material_sum[material] / material_index_sum[material]
                data2[index1][material] = x1
            else:
                data2[index1][material] = temp_sum_value / temp_sum_index
            for or_index, item_origin in enumerate(text_data_origin[place][2]):
                if check(item_origin) == material:
                    temp_or_dit[text_data_origin[place][2][or_index]] = data2[index1][material]
                    # n_ += 1
                    # total_sum += (data2[index1][material] - source_value) ** 2

        count += 1
        write_arr = [text_data_origin[place][0], text_data_origin[place][1]]
        for or_index, info in enumerate(text_data_origin[place][2]):
            if text_data_origin[place][2][or_index] not in temp_or_dit:
                temp_ar = str(info) + '#'
                write_arr.append(temp_ar)
                continue

            temp_ar = str(info) + '#' + str(temp_or_dit[text_data_origin[place][2][or_index]])
            write_arr.append(temp_ar)
        csv_writer.writerow(write_arr)
        place += 1
        if count % 50 == 0:
            print("已成功计算{0}条数据".format(count))
    # PSME = pow(total_sum / n_, 0.5)
    # print(PSME)


if __name__ == '__main__':
    # 调用数据预处理
    print("-------------------")
    print("数据预处理开始:")
    datas1 = dp.data_read('../data/recipe1.csv')  # 训练食谱
    datas3 = dp.data_read('../data/recipe3.csv')  # 测试食谱
    train_data = dp.data_split(datas1)
    text_data = dp.data_split(datas3)
    text_data_origin = dp.data_split2(datas3)
    total_num = dp.get_material_information(train_data)
    dp.clean_data(train_data)
    # dp.normalize(train_data)
    # recipe_normalize_data = train_data
    material_index_probability = dp.material_index_probability
    material_index = dp.material_index
    material_sum = dp.material_sum  # 对应食材标号的食材质量总和
    material_index_sum = dp.material_index_sum  # 单类食材存在个数,用于后续数据填充
    print("数据预处理成功")
    print("-------------------")

    # 食材向量嵌入(使用300维向量)
    print("食材向量化开始:")
    # dp.get_recipe_embedding(total_num, train_data)
    model = Word2Vec(train_data, workers=4, size=material_vec_size, min_count=1, window=10, sample=1e-2, iter=20)
    model.wv.init_sims(replace=True)
    print("食材向量化成功")
    print("-------------------")

    # 食谱向量化
    print("食谱向量化开始:")
    recipe_vec = recipe2vec(train_data)  # 训练食谱向量化
    print("1. 训练食谱向量化成功")
    recipe_vec2 = recipe2vec(text_data)  # 测试食谱向量化
    print("2. 测试食谱向量化成功")
    print("-------------------")

    # 测试食谱结果填补
    print("获得最终结果中...")
    get_ans(recipe_vec, recipe_vec2, train_data, text_data)
    f.close()
    print("文件填写成功")
    print("-------------------")

    # 计算模型均方根误差
    print("模型评测:")
    cP.count_RMSE()
