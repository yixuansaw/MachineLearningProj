# Done By: Saw Yi Xuan (1004655)

import os

# Methods needed for implementation

def get_count_tags(training_set):
    tag_counts = {'START': 0, 'O': 0, 'B-positive': 0, 'B-neutral': 0, 'B-negative': 0, 'I-positive': 0, 'I-neutral': 0, 'I-negative': 0, 'STOP': 0}

    for data_pair in training_set:
        if len(data_pair) > 1:
            tag = data_pair[1]
            if tag in tag_counts:
                tag_counts[tag] += 1
        elif len(data_pair) == 1:
            tag_counts['START'] += 1
            tag_counts['STOP'] += 1

    return tag_counts


def get_tag_counts(training_set):
    tag_counts = {'O': {}, 'B-positive': {}, 'B-neutral': {}, 'B-negative': {},
                       'I-positive': {}, 'I-neutral': {}, 'I-negative': {}}

    for item in training_set:
        if len(item) > 1:
            try:
                label, word = item[1], item[0]
                if word not in tag_counts[label]:
                    tag_counts[label][word] = 1
                else:
                    tag_counts[label][word] += 1
            except IndexError:
                print("error", IndexError, item)
    return tag_counts


def read_path(file_name):
  return os.getcwd() + file_name

def read_dev_data(file_path):
    sentence_list = [[]]
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if line == "\n":
                sentence_list.append([])
            else:
                sentence_list[-1].append(line.strip())
    return sentence_list[:-1]

def read_data(path):
    dataset = []

    with open(path, "r", encoding="utf-8") as f:
        training_set = f.readlines()

    for line in training_set:
        if len(line) == 1:
            dataset.append(["\n"])
        else:
            line = line.rstrip('\n').rsplit(' ', maxsplit=1)
            if len(line) == 2 and line[0] != '' and line[1] != '':
                dataset.append(line)

    return dataset

def extract_unique_words(dataset):
    unique_words = set()
    for entry in dataset:
        if len(entry) > 1:
            unique_words.add(entry[0])
    return unique_words
  

def output_to_file(predictions, input_data, output_path):
    assert len(predictions) == len(input_data)
    output_file = open(output_path, "w", encoding="utf-8")
    num_samples = len(input_data)
    print("Number of Lines: ", num_samples)
    for i in range(num_samples):
        assert len(input_data[i]) == len(predictions[i])
        num_tokens = len(input_data[i])
        for j in range(num_tokens):
            output_file.write(input_data[i][j] + " " + predictions[i][j] + "\n")
        output_file.write("\n")
    print("Output to", output_path)
    print("Number of Lines: ", num_samples)

class EmissionModel:

    # Initialize the EmissionModel instance with k
    def __init__(self, k=1):
        self.k = k
        self.emission_params = {}
        self.labels = ['O', 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative']

    def compute_params(self, count_tags, tag_word_counts):
        for tag, word_counts in tag_word_counts.items():
            total_tag_count = count_tags.get(tag, 0) + self.k
            self.emission_params[tag] = {word: (count + self.k) / total_tag_count for word, count in word_counts.items()}
            self.emission_params[tag]['#UNK#'] = self.k / total_tag_count

    def predict(self, words, word_set):
        return [max(self.labels, key=lambda label: self.emission_params[label].get(word if word in word_set else "#UNK#", 0)) for word in words]

class ComputeScore:
    def get_scores(self, expected, gold):
        correct_sentiment_count = sum(1 for pred_sent, gold_sent in zip(expected, gold) for pred, gold in zip(pred_sent, gold_sent) if pred == gold)
        expected_sentiment_count = sum(1 for pred_sent in expected for pred in pred_sent if pred != 'O')
        gold_sentiment_count = sum(1 for gold_sent in gold for gold in gold_sent if gold != 'O')

        precision = self.get_precision(correct_sentiment_count, expected_sentiment_count)
        recall = self.get_recall(correct_sentiment_count, gold_sentiment_count)
        f_score = self.get_f_score(precision, recall)

        return {'precision': precision, 'recall': recall, 'f_score': f_score}

    def get_precision(self, correct_count, expected_count):
        return correct_count / expected_count if expected_count != 0 else 0

    def get_recall(self, correct_count, gold_count):
        return correct_count / gold_count if gold_count != 0 else 0

    def get_f_score(self, precision, recall):
        return 2 * precision * recall / (precision + recall) if precision != 0 and recall != 0 else 0

def read_dev_gold(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def prediction_loop(separated, compute_e, word_set):
    return [compute_e.predict(doc, word_set) for doc in separated]

def run_emission_prediction(training_path, test_path, output_path):
    train = read_data(training_path)
    train_words = extract_unique_words(train)
    tags = get_count_tags(train)
    test = read_dev_data(test_path)
    tag_words = get_tag_counts(train)

    compute_model = EmissionModel(k=1)
    compute_model.compute_params(tags, tag_words)

    prediction = prediction_loop(test, compute_model, train_words)

    score_calculator = ComputeScore()
    gold = read_dev_gold(test_path)
    scores = score_calculator.get_scores(prediction, gold)
    print("Scores:", scores)

    output_to_file(prediction, test, output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        run_emission_prediction(r"Data/ES/train", r"Data/ES/dev.in", r"Data/ES/dev.p1.out")
        run_emission_prediction(r"Data/RU/train", r"Data/RU/dev.in", r"Data/RU/dev.p1.out")
    elif len(sys.argv) == 4:
        run_emission_prediction(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("usage: python part1.py [train_path] [test_path] [output_path]")

