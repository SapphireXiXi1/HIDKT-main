data_folder = 'dataset'

def get_csv_fname(trainEvalTest, dataset, datasetNum):
    if trainEvalTest == 'train':
        fname = 'train%s.csv'%datasetNum
    elif trainEvalTest == 'eval':
        fname = 'eval%s.csv'%datasetNum
    else:
        fname = 'test%s.csv'%datasetNum
    return '%s/%s/%s' % (data_folder, dataset, fname)


def get_num_skill(dataset):
    if dataset == 'assist0910':
        return 123
    elif dataset == 'assist2017':
        return 410
    elif dataset == 'algebra05':
        return 270
    elif dataset == 'assist2015':
        return 99
    elif dataset == 'assist2012':
        return 264
    else:
        raise NotImplementedError('Invalid Dataset')


def get_num_question(dataset):
    if dataset == 'assist0910':
        return 189202
    elif dataset == 'assist2017':
        return 3161
    elif dataset == 'algebra05':
        return 173112
    elif dataset == 'assist2015':
        return 99
    elif dataset == 'assist2012':
        return 52849
    else:
        raise NotImplementedError('Invalid Dataset')


def read_csv(fname, minimum_seq_len):
    with open(fname, 'r') as f:
        data = f.read()


    data = data.strip().split('\n')

    effLen = []
    questionID = []
    skillID = []
    label = []

    i = 0
    while i < len(data):
        line = data[i]
        if i % 4 == 0:

            if int(line) >= minimum_seq_len:
                effLen.append(int(line))
            else:

                i += 4
                continue
        elif i % 4 == 1:

            line = line.split(',')
            questionID.append([int(e) for e in line if e])
        elif i % 4 == 2:

            line = line.split(',')
            skillID.append([int(e) for e in line if e])
        else:

            line = line.split(',')
            label.append([int(e) for e in line if e])


            if not (effLen[-1] == len(questionID[-1]) == len(skillID[-1]) == len(label[-1])):
                print(f"Skipping invalid record at index {len(effLen) - 1}: "
                      f"effLen={effLen[-1]}, questionID={len(questionID[-1])}, "
                      f"skillID={len(skillID[-1])}, label={len(label[-1])}")

                effLen.pop()
                questionID.pop()
                skillID.pop()
                label.pop()
        i += 1

    return effLen, questionID, skillID, label
