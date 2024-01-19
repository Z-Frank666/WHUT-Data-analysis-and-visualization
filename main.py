# 计算机与人工智能学院 计算机2203班 张俊鑫
# 0122210870531
# 完成日期2023-11-16

from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

file_name = './data/result.csv'

# seaborn 绘图初始化
def sns_init():
    sns.set(style='darkgrid')       # 设置背景
    sns.set(font='SimHei')          # 设置字体，以正确显示中文

# 数据的读取以及数据的预览
def read_and_review(file_name):
    data = pd.read_csv(file_name, delimiter=',',encoding='gbk')  # 利用read_csv函数读取数据
    print(data.head())  #预览
    print(data.isnull().sum(), '\n')  # 查看是否有空缺值
    print(data.shape)  # 打印列表的形状
    print(data.dtypes)  # 检查数据的数据类型是否正确
    return data

# 数据处理-解决各个赛季球星名字翻译出现更迭的问题
# 由于球星翻译更迭属少数个例，直接针对特定属性修改
# 例如：对于多次入选射手榜的B席，还有赛季翻译为全称贝尔纳多席尔瓦，在此以该球员其他属性为索引统一改为B席
# 经观察：出现该情况的球员有：孙兴慜，B席，B费
def data_update(data):
    data.loc[(data['队伍名称']=='热刺')&(data['位置']=='前锋')&(data['国家']=='韩国')&(data['球衣号']=='7号'),['球员姓名']]='孙兴慜'
    data.loc[(data['队伍名称'] == '曼城') & (data['位置'] == '中场') & (data['国家'] == '葡萄牙')&(data['球衣号']=='20号'), ['球员姓名']] = 'B席'
    data.loc[(data['队伍名称'] == '曼联') & (data['位置'] == '中场') & (data['国家'] == '葡萄牙')&(data['球衣号']=='8号'), ['球员姓名']] = 'B费'
    return data

# 绘制2018-2022五个已结束赛季球员进球数散点图
def draw_picture1(data):
    data=data.loc[data['年份'] < 2023]
    sns_init()
    plt.figure(figsize=(9, 5))  # 设置图形大小
    sns.catplot(x="年份",y="总进球数",data=data)
    plt.title("2018-2022球员进球散点图")
    plt.savefig('img/picture1.png')
    plt.show()

# 绘制2018-2022五个已结束赛季球员进球数箱型图
def draw_picture2(data):
    data = data.loc[data['年份'] < 2023]
    sns_init()
    plt.figure(figsize=(9, 5))  # 设置图形大小
    sns.catplot(x="年份", y="总进球数", kind="box", data=data)
    plt.title("2018-2022球员进球箱型图图")
    plt.savefig('img/picture2.png')
    plt.show()

# 获取关于俱乐部进球的关系二维列表
def data_temp(data):
    dict1 = dict() # 定义俱乐部与进球数字典
    for i in range(1,len(data)-50):
        dict1[data[i][1]] = 0
    for i in range(1, len(data) - 50):
        dict1[data[i][1]] += int(data[i][3])
    list1 = []
    for i in dict1:
        list2 = []
        list2.append(i)
        list2.append(dict1[i])
        list1.append(list2)
    return list1

# 绘制2018-2022五个已结束赛季各俱乐部总进球柱形图
def draw_picture3(data):
    sns_init()
    plt.figure(figsize=(8, 4))  # 设置图形大小
    sns.barplot(x='俱乐部', y='总进球数', data=data,palette='hls')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.title('2018-2022五个赛季各俱乐部总进球柱形图')
    plt.xlabel('俱乐部')
    plt.ylabel('总进球数')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("img/picture3.png")
    plt.show()

# 绘制2018-2022五个已结束赛季各俱乐部总进球饼图
def draw_picture4(data,program):
    sns_init()
    plt.figure(figsize=(8, 8))
    sum_ = data['总进球数'].sum()
    plt.axes(aspect=1)
    prog_name = []
    for i in range(len(data)):
        prog_name.append(data.iloc[i,0])
    rank = []
    for i in range(len(data)):
        rank.append(float((data.iloc[i,1]/sum_)*100))
    exp = [0] * len(prog_name)
    num = prog_name.index(program)
    exp[num] = 0.1
    plt.pie(rank, explode=exp, labels=prog_name, labeldistance=1.1,
            autopct='%2.1f%%', shadow=True, startangle=90,
            pctdistance=0.8)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.title('2018-2022五个赛季各俱乐部总进球柱形图')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("img/picture4.png")
    plt.show()

# 获取各国家入选2018-2022每年射手榜前50的人数，分析各国家足球实力
def data_temp2(data):
    dict1 = dict()  # 定义俱乐部与进球数字典
    for i in range(1, len(data) - 50):
        dict1[data[i][7]] = 0
    for i in range(1, len(data) - 50):
        dict1[data[i][7]] += 1
    list1 = []
    for i in dict1:
        list2 = []
        list2.append(i)
        list2.append(dict1[i])
        list1.append(list2)
    return list1

# 选取国家及其入选射手榜人次绘制柱形图
def draw_picture5(data):
    sns_init()
    plt.figure(figsize=(10, 4))
    sns.barplot(x='国家', y='入选射手榜人次', data=data,palette='hls')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.title('2018-2022五个赛季各国家入选英超射手榜人次柱形图')
    plt.xlabel('国家')
    plt.ylabel('入选射手榜人次')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("img/picture5.png")
    plt.show()

# 选取人数排名前10的国家绘制饼图
def draw_picture6(data,nation):
    data = data.iloc[0:10,:]
    sns_init()
    plt.figure(figsize=(8, 8))
    sum_ = data['入选射手榜人次'].sum()
    plt.axes(aspect=1)
    prog_name = []
    for i in range(len(data)):
        prog_name.append(data.iloc[i, 0])
    rank = []  # 百分比的列表
    for i in range(len(data)):
        rank.append(float((data.iloc[i, 1] / sum_) * 100))
    exp = [0] * len(prog_name)
    num = prog_name.index(nation)
    exp[num] = 0.1  # 数据突出显示
    plt.pie(rank, explode=exp, labels=prog_name, labeldistance=1.1,
            autopct='%2.1f%%', shadow=True, startangle=90,
            pctdistance=0.8)
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.title('2018-2022五个赛季入选射手榜人次前十国家柱形图')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("img/picture6.png")
    plt.show()

# 获取2018-2022五赛季均入选射手榜的英超球员信息

# 获取球员入选次数二维列表
def data_temp3(data):
    dict1 = dict()  # 定义俱乐部与进球数字典
    for i in range(1, len(data) - 50):
        dict1[data[i][2]] = 0
    for i in range(1, len(data) - 50):
        dict1[data[i][2]] += 1
    list1 = []
    for i in dict1:
        list2 = []
        list2.append(i)
        list2.append(dict1[i])
        list1.append(list2)
    return list1

# 获取五个赛季均入选的球员名单列表
def get_list(list_temp):
    list2 = []
    for i in list_temp:
        if(i[1]==5):
            list2.append(i[0])
    return list2

# 绘制这些球员进球数折线图
def draw_picture7(data,list_temp):
    data = data.loc[data['年份'] < 2023]
    sns_init()
    list3 = []
    for i in list_temp:
        df_temp = data.loc[data['球员姓名']==i]
        # print(df_temp)
        list3.append(df_temp)
    plt.figure(figsize=(8, 4))  # 设置图形大小
    for i in list3:
        sns.lineplot(x='年份', y='总进球数', data=i, label=i['球员姓名'].iloc[0], marker='o')  # 绘制第一个连载中的折线
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.title('2018-2022连续入选射手榜球员数据折线图')
    plt.xlabel('年份')
    plt.ylabel('总进球数')
    plt.xticks(range(2018,2023),rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("img/picture7")
    plt.show()

# 对萨拉赫，孙兴慜这两位常驻射手做线性回归分析及预测
def get_salahe(data):
    return data.loc[(data['球员姓名']=='萨拉赫')&(data['年份']<=2022)]

def get_sunxinming(data):
    return data.loc[(data['球员姓名']=='孙兴慜')&(data['年份']<=2022)]

def predict_picture(data):#对两位球星2023赛季数据的预测
    sns_init()
    plt.clf()
    color= ['red','blue']
    MSE = []
    R_2 = []
    for k in range(1,4):
        for i in range(2):
            x_1 = data[i][["年份"]]
            y_1 = data[i][["总进球数"]]
            poly_reg = PolynomialFeatures(degree=k)
            x_m = poly_reg.fit_transform(x_1)
            y_m = poly_reg.fit_transform(y_1)
            model_2 = linear_model.LinearRegression()
            model_2.fit(x_m, y_1)
            plt.plot(x_1, model_2.predict(x_m), color=color[i],label = data[i]['球员姓名'].iloc[0])
            plt.legend()
            MSE.append(mean_squared_error(y_1, model_2.predict(x_m)))
            R_2.append(r2_score(y_1, model_2.predict(x_m)))
        print(MSE)#进行曲线拟合的好坏统计
        print(R_2)
        plt.title(f"一元{k}次回归-两位球星赛季进球预测图")
        plt.xlabel('年份')
        plt.ylabel('总进球数')
        plt.xticks(range(2018,2023),rotation=45, ha='right')
        plt.savefig(f'img/{k}_time_predict_picture.png')
        plt.show()

# 绘制球衣号词云——分析顶级射手的球衣号选择
def draw_wordcloud(data,image):
    sns_init()
    graph = np.array(image)  # 图片转数组
    text1 = ' '.join(data['球衣号'])
    wordcloud = WordCloud(font_path='simsun.ttc',  # 中文字体，未指定中文字体时词云的汉字显示为方框，根据系统修改字体名
                   background_color='White',  # 设置背景颜色
                   # background_color=None,   # 设置透明背景
                   # mode='RGBA',
                   mask=graph,  # 设置背景图片
                   max_font_size=240,  # 设置字体最大值
                   scale=1.5).generate(text1)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("顶级射手的球衣号选择词云")
    plt.show()
    wordcloud.to_file("img/wordcloud.png")

# 获取2022年曼城在射手榜上的射手信息，分析曼城在冠军赛季的主要进攻火力分配
def get_temp1(data):
    return data.loc[(data['队伍名称']=='曼城')&(data['年份']==2022)]
#
def draw_picture8(data):
    sns_init()
    plt.figure(figsize=(8, 6))
    colors = ['orange','red']
    sns.barplot(x='球员姓名', y='点球', data=data, color=colors[1], label='点球',
                bottom=data['普通进球'] )
    sns.barplot(x='球员姓名', y='普通进球', data=data, color=colors[0], label='普通进球')
    plt.title('曼城冠军赛季的主要进攻火力')
    plt.xlabel('主攻手')
    plt.ylabel('进球数')
    plt.legend()
    plt.savefig('img/picture8.png')
    plt.show()

if __name__ == '__main__':

    temp = read_and_review(file_name) # 获取dataframe
    temp = data_update(temp)
    narr_temp = np.loadtxt(file_name,delimiter=',',dtype=str)
    narr = temp.to_numpy() # 将dataframe转为numpy便于处理

    draw_picture1(temp)

    draw_picture2(temp)

    list_temp = data_temp(narr)
    df1 =  pd.DataFrame(list_temp, columns=['俱乐部', '总进球数'])  # 将列表整理为dataframe

    draw_picture3(df1)

    program = '曼城'
    draw_picture4(df1,program)

    list_temp2 = data_temp2(narr)
    list_temp2=sorted(list_temp2,key=lambda x:x[1],reverse=True)
    df2 = pd.DataFrame(list_temp2,columns=['国家','入选射手榜人次']) # 将列表整理为dataframe

    draw_picture5(df2)

    nation = '英国'
    draw_picture6(df2,nation)

    list_temp3 = data_temp3(narr)
    player_list = get_list(list_temp3)

    draw_picture7(temp,player_list)

    list_temp4 = []
    list_temp4.append(get_salahe(temp))
    list_temp4.append(get_sunxinming(temp))
    predict_picture(list_temp4)

    images = Image.open("./data/round.jpg")
    draw_wordcloud(temp,images)

    draw_picture8(get_temp1(temp))









