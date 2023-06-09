import os
import json
import numpy 

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

DATAPATH = "./Korean_QA/data"
file = ["KMQA_train4.json", "KMQA_dev4.json"]

#train : file[0]
#dev : file[1]
filelocation = os.path.join(DATAPATH, file[1])

with open(filelocation, 'r') as json_file:
  data = json.load(json_file)['data']

cont_lens = []
context_lens = []
q_lens = []
qu_lens = []
a_lens = []
an_lens = []
answer_positions = []
para_num = 0
cont_num = 0
q_num = 0

for x in range(len(data)):
  paragraphs = data[x]['paragraphs']
  para_num += 1

  ## context
  for para in paragraphs:
    context = para['context']
    cont_num += 1
    con_tokens = tokenizer.tokenize(context)
    cont_lens.append(len(con_tokens))
    context_lens.append(len(context))
    ## qas
    qas = para['qas']
    for qa in qas:
      question = qa['question']
      answer = qa['answers'][0]['text']
      answer_position = qa['answers'][0]['answer_start']
      q_num += 1
      q_tokens = tokenizer.tokenize(question)
      a_tokens = tokenizer.tokenize(answer)

      qu_lens.append(len(question))
      an_lens.append(len(answer))

      q_lens.append(len(q_tokens))
      a_lens.append(len(a_tokens))
      answer_positions.append(answer_position)

print(f"unique paragraphs : {para_num}")
print(f"context number : {cont_num}")
print(f"question number : {q_num}")
print("-------------------------------------------")
print("Number of Tokens")
print(f"Avg paragraph length : {sum(cont_lens)/len(cont_lens)}")
print(f"Avg question length : {sum(q_lens)/len(q_lens)}")
print(f"Avg answer length : {sum(a_lens)/len(a_lens)}")
print("-------------------------------------------")
print(f"Avg answer position : {sum(answer_positions)/len(answer_positions)}")
print("-------------------------------------------")
print("Number of Characters")
print(f"Avg paragraph length : {sum(context_lens)/len(context_lens)}")
print(f"Avg question length : {sum(qu_lens)/len(qu_lens)}")
print(f"Avg answer length : {sum(an_lens)/len(an_lens)}")
