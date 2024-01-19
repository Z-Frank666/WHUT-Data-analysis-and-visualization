import requests
import json
import csv
import os
from scrapy.selector import Selector

date_list = []  # 存储数据的列表


def get_data(response_date):
    """
    从响应数据中获取数据
    参数：
    response_date -- 响应数据
    返回：
    data -- 解析后的数据
    """
    response_text = response_date.text
    start_index = response_text.find('(') + 1
    end_index = response_text.rfind(')')
    json_data = response_text[start_index:end_index]
    # 使用 json.loads() 方法解析 JSON 数据
    data = json.loads(json_data)
    return data['result']['data']


def deep_request(player_id):
    """
    深度请求，获取球员详细信息
    参数：
    player_id -- 球员id
    返回：
    deep_list -- 球员详细信息列表
    """
    url = f'''http://match.sports.sina.com.cn/football/player.php?id={player_id}'''  # 设置url链接
    response = requests.get(url=url)  # 发送请求获取响应数据
    sel = Selector(text=response.content.decode('gbk'))  # 这是一次编码的过程
    sel_data_location = sel.xpath('/html/body/div/div[3]/div[2]/div[1]/div/div[2]/dl[2]/dd[2]/text()').extract_first()
    sel_data_nation = sel.xpath('/html/body/div/div[3]/div[2]/div[1]/div/div[2]/dl[2]/dd[3]/text()').extract_first()
    sel_data_jersey = sel.xpath('/html/body/div/div[3]/div[2]/div[1]/div/div[2]/dl[2]/dd[5]/text()').extract_first()
    # print(f'位置   :{sel_data_location}')
    # print(f'国家   :{sel_data_nation}')
    # print(f'球衣号 :{sel_data_jersey}')
    deep_list = [sel_data_location, sel_data_nation, sel_data_jersey]  # 生成新的列表 并返回
    print(deep_list)
    return deep_list


def store_data(date_list):
    """
    存储数据到csv文件
    参数：
    date_list -- 数据列表
    """
    csv_header = ['年份', '队伍名称', '球员姓名', '总进球数', '普通进球', '点球', '位置', '国家', '球衣号']
    is_empty = not os.path.isfile('./data/result.csv') or os.stat('./data/result.csv').st_size == 0  # 判断文件是否为空，防止表头重复输入

    with open('./data/result.csv', 'a', newline='', encoding='gbk') as file:
        writer = csv.writer(file)
        if is_empty:
            writer.writerow(csv_header)

        for item_list in date_list:
            writer.writerow(item_list)
    print('写入结束')


if __name__ == '__main__':

    for year in range(2018, 2024):  # 循环遍历年份从2018到2023+1
        date_list.clear()
        temp_url = f'''
            https://api.sports.sina.com.cn/?p=sports&s=sport_client&a=index&_sport_t_=football&_sport_s_=opta&_sport_a_=playerorder&item=13&type=4&season={year}&limit=50&callback=CB_979D0953_6B28_4683_BAE1_D907D42D808C1&dpc=1
            '''
        response_data = requests.get(url=temp_url)  # 发送请求获取响应数据
        data_list = get_data(response_data)  # 解析响应数据并存储到data_list列表中
        for item in data_list:
            temp_list = [item.get('year'), item.get('team_name'), item.get('player_name'), item.get('item1'),
                         item.get('item2'), item.get('item3')]
            temp_list.extend(deep_request(item.get('player_id')))  # 将新生成的列表从尾部加入列表中
            date_list.append(temp_list)
        print(date_list)  # 调试代码
        store_data(date_list)
