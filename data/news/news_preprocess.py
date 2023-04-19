import re
import os
import gensim
import numpy as np

PATTERN1 = re.compile('[^A-Za-z]')  # 大小写字母
PATTERN2 = re.compile('[ ]{2,}')  # TODO 啥模式 #{2,} 匹配2个前面表达式
WORD_FREQ_THRESHOLD = 2 # 词频阈值
ENTITY_FREQ_THRESHOLD = 1 # 实体频率阈值
MAX_TITLE_LENGTH = 10  # 标题最大长度
WORD_EMBEDDING_DIM = 50  # 词嵌入维度

word2freq = {}  # 词频
entity2freq = {}
word2index = {}  # 词频超过阈值的编号
entity2index = {}
corpus = []


def count_word_and_entity_freq(files):
    """
    Count the frequency of words and entities in news titles in the training and test files
    :param files: [training_file, test_file]
    :return: None
    """
    for file in files:  # 遍历文件
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.strip().split('\t')
            news_title = array[1]  # 新闻标题
            entities = array[3]  # 新闻标题中的实体

            # count word frequency
            for s in news_title.split(' '):
                if s not in word2freq:
                    word2freq[s] = 1
                else:
                    word2freq[s] += 1

            # count entity frequency
            for s in entities.split(';'):
                entity_id = s[:s.index(':')]
                if entity_id not in entity2freq:  # 实体出现频率
                    entity2freq[entity_id] = 1
                else:
                    entity2freq[entity_id] += 1

            corpus.append(news_title.split(' ')) # 所有新闻标题组成一个大的list
        reader.close()


def construct_word2id_and_entity2id():
    """
    Allocate each valid word and entity a unique index (start from 1)
    :return: None
    """
    cnt = 1  # 0 is for dummy word  dummy->虚假的
    for w, freq in word2freq.items():
        if freq >= WORD_FREQ_THRESHOLD:
            word2index[w] = cnt  #给一个编号
            cnt += 1 # 编号+1
    print('- word size: %d' % len(word2index)) # 超过词频阈值 词的数量

    writer = open('../kg/entity2index.txt', 'w', encoding='utf-8')  # 写文件 实体到下标
    cnt = 1
    for entity, freq in entity2freq.items():
        if freq >= ENTITY_FREQ_THRESHOLD:
            entity2index[entity] = cnt
            writer.write('%s\t%d\n' % (entity, cnt))  # for later use
            cnt += 1
    writer.close()
    print('- entity size: %d' % len(entity2index))


def get_local_word2entity(entities):
    """
    Given the entities information in one line of the dataset, construct a map from word to entity index
    E.g., given entities = 'id_1:Harry Potter;id_2:England', return a map = {'harry':index_of(id_1),
    'potter':index_of(id_1), 'england': index_of(id_2)}
    :param entities: entities information in one line of the dataset
    :return: a local map from word to entity index
    """
    local_map = {}

    for entity_pair in entities.split(';'):
        entity_id = entity_pair[:entity_pair.index(':')]
        entity_name = entity_pair[entity_pair.index(':') + 1:]

        # remove non-character word and transform words to lower case
        # // 两个正则匹配，第一个是只匹配字符 第二个将之前的变成小写
        entity_name = PATTERN1.sub(' ', entity_name)
        entity_name = PATTERN2.sub(' ', entity_name).lower()

        # constructing map: word -> entity_index
        for w in entity_name.split(' '): # 一个实体可能有多个词
            entity_index = entity2index[entity_id]
            local_map[w] = entity_index  # TODO 这里有可能覆盖写

    return local_map


def encoding_title(title, entities):
    """
    Encoding a title according to word2index map and entity2index map
    :param title: a piece of news title
    :param entities: entities contained in the news title
    :return: encodings of the title with respect to word and entity, respectively
    """
    local_map = get_local_word2entity(entities) # 单词到实体的映射
    # TODO 这里有问题 为什么不是全局的

    array = title.split(' ')
    word_encoding = ['0'] * MAX_TITLE_LENGTH
    entity_encoding = ['0'] * MAX_TITLE_LENGTH

    point = 0
    for s in array:
        if s in word2index:
            word_encoding[point] = str(word2index[s])
            if s in local_map:  # 有就映射 没有就是0
                entity_encoding[point] = str(local_map[s])
            point += 1
        if point == MAX_TITLE_LENGTH:
            break
    word_encoding = ','.join(word_encoding) # 变成字符串
    entity_encoding = ','.join(entity_encoding)
    return word_encoding, entity_encoding


def transform(input_file, output_file):
    """
    原文件的标题 和实体 都变成
    :param input_file:
    :param output_file:
    :return:
    """
    reader = open(input_file, encoding='utf-8')
    writer = open(output_file, 'w', encoding='utf-8')
    for line in reader:
        array = line.strip().split('\t')
        user_id = array[0]
        title = array[1]
        label = array[2]
        entities = array[3]
        word_encoding, entity_encoding = encoding_title(title, entities)
        writer.write('%s\t%s\t%s\t%s\n' % (user_id, word_encoding, entity_encoding, label))
    reader.close()
    writer.close()


def get_word2vec_model():
    if not os.path.exists('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model'):
        print('- training word2vec model...')
        w2v_model = gensim.models.Word2Vec(corpus, vector_size=WORD_EMBEDDING_DIM, min_count=1, workers=16)
        print('- saving model ...')
        w2v_model.save('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    else:
        print('- loading model ...')
        w2v_model = gensim.models.word2vec.Word2Vec.load('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    return w2v_model


if __name__ == '__main__':
    print('counting frequencies of words and entities ...')
    count_word_and_entity_freq(['raw_train.txt', 'raw_test.txt'])  # 统计词频和实体频率

    print('constructing word2id map and entity to id map ...')
    construct_word2id_and_entity2id()  # 词编号 实体编号

    print('transforming training and test dataset ...')
    # 构建训练测试数据
    transform('raw_train.txt', 'train.txt')
    transform('raw_test.txt', 'test.txt')

    print('getting word embeddings ...')
    embeddings = np.zeros([len(word2index) + 1, WORD_EMBEDDING_DIM]) # +1 因为有一个默认值
    model = get_word2vec_model()
    for index, word in enumerate(word2index.keys()):
        # TODO 修改接口
        embedding = model.wv[word] if word in model.wv else np.zeros(WORD_EMBEDDING_DIM)
        # embedding = model[word] if word in model.wv.vocab else np.zeros(WORD_EMBEDDING_DIM)
        embeddings[index + 1] = embedding
    print('- writing word embeddings ...')
    # 保存 word embedding
    np.save(('word_embeddings_' + str(WORD_EMBEDDING_DIM)), embeddings)
