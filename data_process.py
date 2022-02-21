import pandas as pd
import os
from tqdm import tqdm
import json


def news():
    paths = os.listdir('文本分类数据集/ClassFile')

    label = []
    text = []

    for path in tqdm(paths):
        if path == '.DS_Store':
            continue
        p = '文本分类数据集/ClassFile/' + path

        file_path = os.listdir(p)
        for file in file_path:
            with open(p + '/' + file, encoding='GBK', errors='ignore') as f:
                content = f.read()
                content = content.strip()
                content = content.split('。')[0] + '。'
                content = content.replace('\n', '')
                content = content.replace('\t', '')
                content = content.replace(' ', '')
                content = content.replace('&nbsp', '')
                while content[0] == ';':
                    content = content[1:]
                content = content.strip()
                if 10 < len(content) < 128:
                    label.append(path)
                    text.append(content)

    df = pd.DataFrame({'label': label, 'text': text})
    df.to_csv('data/news_10.csv', index=False, encoding='utf_8_sig')


def sentiment():
    text = []
    label = []
    for path in os.listdir('文本分类数据集/corpus/neg'):
        with open('文本分类数据集/corpus/neg/' + path, encoding='GBK', errors='ignore') as file:
            content = file.read()
            content = content.replace('\n', '')
            content = content.replace('\t', '')
            content = content.replace(' ', '')
            if content:
                text.append(content)
                label.append(0)

    for path in os.listdir('文本分类数据集/corpus/pos'):
        with open('文本分类数据集/corpus/pos/' + path, encoding='GBK', errors='ignore') as file:
            content = file.read()
            content = content.replace('\n', '')
            content = content.replace('\t', '')
            content = content.replace(' ', '')
            if content:
                text.append(content)
                label.append(1)

    df = pd.DataFrame({'label': label, 'text': text})
    df.to_csv('data/sentiment_hotel.csv', index=False, encoding='utf_8_sig')


def sentiment2():
    df1 = pd.read_csv('文本分类数据集/sentiment/train.tsv', sep='\t')
    df2 = pd.read_csv('文本分类数据集/sentiment/dev.tsv', sep='\t')
    df3 = pd.read_csv('文本分类数据集/sentiment/test.tsv', sep='\t')

    df = pd.concat([df1, df2, df3])
    df.rename(columns={'text_a': 'text'}, inplace=True)
    df.to_csv('data/sentiment.csv', index=False, encoding='utf_8_sig')


def fudan_news():
    label = []
    text = []

    for p_path in ['answer', 'train']:

        paths = os.listdir('文本分类数据集/Fudan/' + p_path)
        for path in tqdm(paths):
            if path == '.DS_Store':
                continue
            p = '文本分类数据集/Fudan/answer/' + path + '/utf8/'

            file_path = os.listdir(p)
            for file in file_path:
                with open(p + '/' + file, encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if '【 标  题 】' in content:
                        content = content.split('【 标  题 】')[1]
                        content = content.split('\n')[0]
                        content = content.strip()

                        if content:
                            label.append(path)
                            text.append(content)

    df = pd.DataFrame({'label': label, 'text': text})
    df.to_csv('data/fudan_news.csv', index=False, encoding='utf_8_sig')


def iflytek():
    label = []
    text = []
    with open('文本分类数据集/iflytek_public/train.json') as file:
        for line in file.readlines():
            j = json.loads(line.strip())
            label.append(int(j['label']))
            text.append(j['sentence'])
    with open('文本分类数据集/iflytek_public/dev.json') as file:
        for line in file.readlines():
            j = json.loads(line.strip())
            label.append(int(j['label']))
            text.append(j['sentence'])
    df = pd.DataFrame({'label': label, 'text': text})
    df.to_csv('data/iflytek.csv', index=False, encoding='utf_8_sig')


def short_news():
    df1 = pd.read_csv('文本分类数据集/shorttext_zj/train.csv')
    df2 = pd.read_csv('文本分类数据集/shorttext_zj/dev.csv')
    df3 = pd.read_csv('文本分类数据集/shorttext_zj/test.csv')

    df = pd.concat([df1, df2, df3])
    df.rename(columns={'sentence': 'text'}, inplace=True)
    df.to_csv('data/tnews.csv', index=False, encoding='utf_8_sig')


def weibo():
    df = pd.read_csv('文本分类数据集/weibo/simplifyweibo_4_moods.csv')
    df.rename(columns={'review': 'text'}, inplace=True)
    df.to_csv('data/weibo.csv', index=False, encoding='utf_8_sig')


def hotel():
    df1 = pd.read_csv('data/sentiment_hotel.csv')
    df2 = pd.read_csv('data/sentiment.csv')
    df = df1.append(df2)
    df.drop_duplicates(inplace=True)
    df.to_csv('data/hotel_sentiment.csv', index=False, encoding='utf_8_sig')


def thucnews():
    p_path = '/Users/joezhao/Downloads/THUCNews/'

    label = []
    text = []

    for path in tqdm(os.listdir(p_path)):
        if path == '.DS_Store':
            continue
        p = p_path + path

        file_path = os.listdir(p)
        for file in file_path:
            with open(p + '/' + file, encoding='utf-8', errors='ignore') as f:
                content = f.read()

                content = content.split('\n')[0]
                content = content.strip()

                if content:
                    label.append(path)
                    text.append(content)
    df = pd.DataFrame({'label': label, 'text': text})
    df.to_csv('data/thucnews.csv', index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    thucnews()
