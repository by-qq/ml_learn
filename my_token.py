import keras.src.layers
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 构建分词用keras的t，但是需要数据预处理（换行符换成<eos>）
def data_processing_tokenization(text):
    processing_data = []
    with open(text,"r",encoding="utf-8") as f:
        # read是一个元素，readlines是一行一个元素,读出来就自动转换成了列表
        data = f.readlines()
    for line in data:
        line = line.replace("\n"," <eos>")  # 注意空格
        processing_data.append(line)
    # print(data,"\n",processing_data)

    textvector = layers.TextVectorization(
        max_tokens = 2000,
        split  = "whitespace",
        output_mode = "int",
    )
    textvector.adapt(processing_data)
    result_data = textvector.get_vocabulary()
    word_dict = dict(zip(result_data, range(len(result_data))))

    # print(processing_data,"\n",textvector,"\n",result_data)
    return processing_data,textvector,result_data

# 构建序列预测数据集,这里需要将文字转化成int了
def build_dataset(processing_data,textvector,window_size=2):
    data = []
    target = []
    for line in processing_data:
        # 将文本转换为整数序列
        sequence = textvector(line)
        # print(sequence)
        sequence = sequence.numpy()
        # print(sequence)
        if len(sequence) > window_size:
            for i in range(len(sequence) - window_size):
                data.append(sequence[i:i+window_size])
                target.append(sequence[i+window_size])
    data_tensor = tf.constant(data, dtype=tf.int32)
    target_tensor = tf.constant(target, dtype=tf.int32)
    # print(data, "\n", target)
    dataset = tf.data.Dataset.from_tensor_slices((data_tensor, target_tensor))
    print(dataset)
    return dataset

# 构建模型
def build_model(vocab_size):
    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=32,
            input_length=2,
        ),
        layers.GRU(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(vocab_size, activation='softmax')
    ])
    # 编译模型
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def predict_next_word(model, vectorizer, input_text, vocab):

    # 预处理输入文本，去除两边的空值，并转化成序列
    processed_text = input_text.strip()
    sequence = vectorizer([processed_text])
    sequence = sequence.numpy()[0]  # 维度裁剪

    # 如果序列长度超过窗口大小，只取最后window_size个词，根据这几个词进行预测
    if len(sequence) > 2:
        sequence = sequence[-2:]

    # 如果序列长度不足，进行填充（在前面填充0），不在后边填充是为了防止将填充值认为是有效填充
    if len(sequence) < 2:
        padding = [0] * (2 - len(sequence))
        sequence = padding + sequence.tolist()

    # 预测
    input_seq = tf.constant([sequence], dtype=tf.int32)
    predictions = model.predict(input_seq, verbose=0)

    # 获取最可能的下一个词
    next_word_idx = np.argmax(predictions[0])
    next_word = vocab[next_word_idx]

    return next_word


# 6. 循环文本生成
def generate_text(model, vectorizer, prompt, vocab, max_length=10):
    """根据提示词循环生成文本"""
    generated_text = prompt
    current_input = prompt

    print(f"\n开始文本生成:")
    print(f"初始提示: '{prompt}'")

    # 如果当前输入太长，只保留最后window_size个词
    words = current_input.split()
    if len(words) > 2:
        current_input = ' '.join(words[-2:])

    for i in range(max_length):
        # 预测下一个词
        next_word = predict_next_word(model, vectorizer, current_input, vocab)
        print(f"步骤 {i + 1}: 当前输入 '{current_input}' -> 预测下一个词: '{next_word}'")

        # 如果遇到<eos>或者未知词，停止生成
        if next_word == 'eos' or next_word == '[UNK]':
            print(f"遇到终止符 '{next_word}'，停止生成")
            break

        # 更新生成的文本和当前输入
        generated_text += ' ' + next_word
        current_input += ' ' + next_word



    print(f"\n最终生成的文本: '{generated_text}'")
    return generated_text


if __name__ == "__main__":

    data_path = r"C:\Users\28021\PycharmProjects\PythonProject\text_cls\corpus.txt"
    data = "深度 学习"

    processing_data,textvector,result_data = data_processing_tokenization(data_path)

    dataset = build_dataset(processing_data,textvector)
    dataset = dataset.shuffle(100).batch(8).prefetch(tf.data.AUTOTUNE)

    model = build_model(len(result_data))
    model.fit(dataset,epochs = 50 ,verbose = 1)

    test_prompt = "深度 学习"
    next_word = predict_next_word(model, textvector, test_prompt, result_data)
    print(f"\n单次预测结果:")
    print(f"输入: '{test_prompt}' -> 预测下一个词: '{next_word}'")

    # 循环生成
    generated_text = generate_text(
        model, textvector, "我 喜欢", result_data, max_length=6
    )










