import tensorflow as tf
import numpy as np
import random

def main():

    label_index = {
        'aardvark': 0,
        'antelope': 1,
        'bass': 2,
        'bear': 3,
        'boar': 4,
        'buffalo': 5,
        'calf': 6,
        'carp': 7,
        'catfish': 8,
        'cavy': 9,
        'cheetah': 10,
        'chicken': 11,
        'chub': 12,
        'clam': 13,
        'crab': 14,
        'crayfish': 15,
        'crow': 16,
        'deer': 17,
        'dogfish': 18,
        'dolphin': 19,
        'dove': 20,
        'duck': 21,
        'elephant': 22,
        'flamingo': 23,
        'flea': 24,
        'frog': 25,
        'fruitbat': 26,
        'giraffe': 27,
        'girl': 28,
        'gnat': 29,
        'goat': 30,
        'gorilla': 31,
        'gull': 32,
        'haddock': 33,
        'hamster': 34,
        'hare': 35,
        'hawk': 36,
        'herring': 37,
        'honeybee': 38,
        'housefly': 39,
        'kiwi': 40,
        'ladybird': 41,
        'lark': 42,
        'leopard': 43,
        'lion': 44,
        'lobster': 45,
        'lynx': 46,
        'mink': 47,
        'mole': 48,
        'mongoose': 49,
        'moth': 50,
        'newt': 51,
        'octopus': 52,
        'opossum': 53,
        'oryx': 54,
        'ostrich': 55,
        'parakeet': 56,
        'penguin': 57,
        'pheasant': 58,
        'pike': 59,
        'piranha': 60,
        'pitviper': 61,
        'platypus': 62,
        'polecat': 63,
        'pony': 64,
        'porpoise': 65,
        'puma': 66,
        'pussycat': 67,
        'raccoon': 68,
        'reindeer': 69,
        'rhea': 70,
        'scorpion': 71,
        'seahorse': 72,
        'seal': 73,
        'sealion': 74,
        'seasnake': 75,
        'seawasp': 76,
        'skimmer': 77,
        'skua': 78,
        'slowworm': 79,
        'slug': 80,
        'sole': 81,
        'sparrow': 82,
        'squirrel': 83,
        'starfish': 84,
        'stingray': 85,
        'swan': 86,
        'termite': 87,
        'toad': 88,
        'tortoise': 89,
        'tuatara': 90,
        'tuna': 91,
        'vampire': 92,
        'vole': 93,
        'vulture': 94,
        'wallaby': 95,
        'wasp': 96,
        'wolf': 97,
        'worm': 98,
        'wren': 99
    }

    data = []
    labels = []

    with open("zoo.data", "r") as input_file:
        for line in input_file:
            if(len(line.strip()) == 0):
                continue

            full_data_line = line.strip().split(",")

            data_line = full_data_line[1:18]
            label_line = full_data_line[0]

            data_line = list(map(float, data_line))

            if label_line in label_index:
                label_line = label_index[label_line]
            else:
                print("Bad data", line)
                continue

            data.append(data_line)
            labels.append(label_line)



    #print("data", len(data))
    #print("labels", len(labels))

    dataset = list(zip(data, labels))
    random.shuffle(dataset)
    test_length = int(len(dataset) * 0.67)

    #print("test_length", test_length)
    train_dataset = dataset[:test_length]
    test_dataset = dataset[test_length:]

    x_size = 17
    output_size = 100
    num_nodes = 250

    # Symbols
    inputs = tf.placeholder("float", shape=[None, x_size])
    labels = tf.placeholder("int32", shape=[None])

    weights1 = tf.get_variable("weight1", shape=[x_size, num_nodes], initializer=tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable("bias1", shape=[num_nodes], initializer=tf.constant_initializer(value=0.0))

    layer1 = tf.nn.relu(tf.matmul(inputs, weights1) + bias1)

    weights2 = tf.get_variable("weight2", shape=[num_nodes, num_nodes], initializer=tf.contrib.layers.xavier_initializer())
    bias2 = tf.get_variable("bias2", shape=[num_nodes], initializer=tf.constant_initializer(value=0.0))

    layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + bias2)

    weights3 = tf.get_variable("weight3", shape=[num_nodes, output_size], initializer=tf.contrib.layers.xavier_initializer())
    bias3 = tf.get_variable("bias3", shape=[output_size], initializer=tf.constant_initializer(value=0.0))

    outputs = tf.matmul(layer2, weights3) + bias3

    # backprop
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, 100), logits=outputs))
    train = tf.train.AdamOptimizer().minimize(loss)

    predictions = tf.argmax(tf.nn.softmax(outputs), axis=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        previous_test_loss_output, previous_test_prediction_output = 999999, 0.0
        loop = True

        #for epoch in range(500):
        while loop:
            batch = random.sample(train_dataset, 50)
            inputs_batch, labels_batch = zip(*batch)
            loss_output, prediction_output, _ = sess.run([loss, predictions, train], feed_dict={inputs: inputs_batch, labels: labels_batch})

            #print("prediction_output", prediction_output)
            #print("labels_batch", labels_batch)
            #accuracy = np.mean(labels_batch == prediction_output)
            #print("train", "loss", loss_output, "accuracy", accuracy)

            batch = random.sample(test_dataset, 30)
            test_inputs_batch, test_labels_batch = zip(*batch)
            test_loss_output, test_prediction_output = sess.run([loss, predictions], feed_dict={inputs: test_inputs_batch, labels: test_labels_batch})

            if previous_test_loss_output < test_loss_output:
                loop = False

            previous_test_loss_output, previos_test_prediction_output = test_loss_output, test_prediction_output

            print("test_prediction_output", test_prediction_output)
            print("test_labels_batch", labels_batch)
            accuracy = np.mean(test_labels_batch == test_prediction_output)
            print("test", "loss", test_loss_output, "accuracy", accuracy)


if __name__ == "__main__":
    main()
