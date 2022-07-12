import json
from conllu import parse
from conllu import parse_incr
import conllu

types = {
    'B-PER': 0,
    'B-LOC': 1,
    'B-ORG': 2,
}

train_size = 500
dev_size = 250

# * total mention count: 1175
# * total mention count: 1235
# * total mention count: 1256
# * total mention count: 1262
# * total mention count: 1460
# * total mention count: 2081

for year in range(2014, 2020):
    file = open(f"/Users/danielk/ideaProjects/temporal_drift/ner/conll/{year}.conll", 'r')
    outfile_train = open(f"/Users/danielk/ideaProjects/temporal_drift/ner/out/train/indiv/{year}.train.indivs.forlabels.scores", 'w+')
    outfile_dev = open(f"/Users/danielk/ideaProjects/temporal_drift/ner/out/dev/{year}.dev.forlabels.scores", 'w+')
    outfile_test = open(f"/Users/danielk/ideaProjects/temporal_drift/ner/out/test/{year}.test.forlabels.scores", 'w+')
    total_mention_count = 0
    tokens_thus_far = ""
    mention_thus_far = ""
    mentions = []
    mention_tags = []
    for line in file.readlines():
        line_split = line.replace("\n", "").split(" ")
        if len(line_split) == 1:
            # print(" - - - - ")
            # print(tokens_thus_far)
            # print(mentions)
            # print(mention_tags)
            if tokens_thus_far != '':
                for mention, tag in zip(mentions, mention_tags):
                    row = {
                        'text': f"{mention} [SEP] {tokens_thus_far}",
                                              'labels': types[tag],
                        'year': year
                    }
                    if total_mention_count <= train_size:
                        outfile_train.write(json.dumps(row) + "\n")
                    elif total_mention_count <= train_size + dev_size:
                        outfile_dev.write(json.dumps(row) + "\n")
                    else:
                        outfile_test.write(json.dumps(row) + "\n")

                tokens_thus_far = ''
                mention_thus_far = ''
                mentions = []
                mention_tags = []
            continue
        token = line_split[0]
        tag = line_split[1]
        tokens_thus_far += token + " "
        if "B-" in tag:
            mention_thus_far += token + " "
            mention_tags.append(tag)
            total_mention_count += 1
        elif "O" == tag:
            if mention_thus_far != "":
                mentions.append(mention_thus_far)
                mention_thus_far = ""
    print(f" * total mention count: {total_mention_count}")

