import torch
import transformers
from gensim.models import FastText
from torch.optim.lr_scheduler import StepLR
from transformers import RobertaTokenizer, RobertaModel,BertModel, BertTokenizer,GPT2Model, GPT2Tokenizer, DistilBertModel, DistilBertTokenizer, OPTModel, AutoTokenizer, T5Tokenizer, T5Model, T5EncoderModel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os 
from cbm_template_models import MLP, FC
from cbm_models import ModelXtoC_function, ModelCtoY_function,ModelXtoCtoY_function

from sparse_model import insert_sparse_mask, set_required_gradient, set_sparse_index, setup_seed, SubnetLinear_Mask
from obert import EmpiricalBlockFisherInverse
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--VERBOSE', action='store_true', help='Enable VERBOSE logs.')
parser.add_argument('--model_name', default='distilbert-base-uncased',
                     help='Backbone model.')
parser.add_argument('--BATCH_SIZE', type=int, default=8,
                    help='Value of BLOCK_SIZE.')
parser.add_argument('--FINETUNE_EPOCH', type=int, default=30,
                    help='Number of epochs to train the model.')
parser.add_argument('--START_EPOCH', type=int, default=2,
                    help='Number of epochs to start pruning.')
parser.add_argument('--END_EPOCH', type=int, default=15,
                    help='Number of epochs to end pruning.')
parser.add_argument('--CLF_EPOCH', type=int, default=20,
                    help='Maximum Number of epochs to train the clf head.')
parser.add_argument('--INIT_SPARSITY', type=float, default=0.2,
                    help='Rate for the initial target sparsity.')
parser.add_argument('--FINAL_SPARSITY', type=float, default=0.75,
                    help='Rate for the final target sparsity.')
parser.add_argument('--M_GRAD', type=int, default=20,
                    help='Values of M_GRAD.')
parser.add_argument('--BLOCK_SIZE', type=int, default=50,
                    help='Value of BLOCK_SIZE.')
parser.add_argument('--NUM_LAYERS', type=int, default=20,
                    help='Number of layers masked each time.')
parser.add_argument('--LAMBD', type=float, default=1e-7,
                    help='Values of LAMBD.')
parser.add_argument('--LAMBD_XtoC', type=float, default=5.,
                    help='Rate for the final target sparsity.')
parser.add_argument('--model_path', default='/scratch/ztan36/s4e_model/',
                     help='Path for output pre-trained model.')

args = parser.parse_args()  

model_path = args.model_path


# Enable concept or not
mode = 'joint'

# Define the paths to the dataset and pretrained model
# model_name = "microsoft/deberta-base"
model_name = args.model_name # 'bert-base-uncased' / 'roberta-base' / 'gpt2' / 'lstm' / 'distilbert-base-uncased'
setup_seed(args.seed)

# Define the maximum sequence length and batch size
max_len = 512
batch_size = args.BATCH_SIZE
lambda_XtoC = args.LAMBD_XtoC  # lambda > 0
is_aux_logits = False
num_labels = 5  #label的个数
num_each_concept_classes = 3  #每个concept有几个类


# Load the tokenizer and pretrained model
if model_name == 't5-base':
    model_type = 't5'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
    model = T5EncoderModel.from_pretrained("t5-base")
elif model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model_type = 'SLM'
elif model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model_type = 'SLM'
elif model_name == 'distilbert-base-uncased':
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    model_type = 'SLM'
elif model_name == 'gpt2':
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    model_type = 'SLM'
elif model_name == 'facebook/opt-125m':
    model = OPTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_type = 'OPT-125m'
elif model_name == 'facebook/opt-350m':
    model = OPTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_type = 'OPT-350m'
elif model_name == 'facebook/opt-1.3b':
    model = OPTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_type = 'OPT-1.3b'
elif model_name == 'lstm':
    fasttext_model = FastText.load_fasttext_format('./fasttext/cc.en.300.bin')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_type = 'LSTM'
    

    class BiLSTMWithDotAttention(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            embeddings = fasttext_model.wv.vectors
            self.embedding.weight = torch.nn.Parameter(torch.tensor(embeddings))
            self.embedding.weight.requires_grad = False
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers = 1, bidirectional=True, batch_first=True)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim*2, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
        )

        def forward(self, input_ids, attention_mask):
            input_lengths = attention_mask.sum(dim=1)
            embedded = self.embedding(input_ids)
            output, _ = self.lstm(embedded)
            weights = F.softmax(torch.bmm(output, output.transpose(1, 2)), dim=2)
            attention = torch.bmm(weights, output)
            logits = self.classifier(attention.mean(1))
            return logits

    model = BiLSTMWithDotAttention(len(tokenizer.vocab), 300, 128)

data_type = "pure_cebab" # "pure_cebab"/"aug_cebab"/"aug_yelp"/"aug_cebab_yelp"
# Load data
if data_type == "pure_cebab":
    num_concept_labels = 4
    train_split = "train_exclusive"
    test_split = "test"
    val_split = "validation"
    CEBaB = load_dataset("CEBaB/CEBaB")
    # CEBaB = {}
    # CEBaB[train_split] = pd.read_csv("../../dataset/CEBaB-v1.1/train_exclusive.json")
    # CEBaB[test_split] = pd.read_csv("../../dataset/CEBaB-v1.1/test.json")
    # CEBaB[val_split] = pd.read_csv("../../dataset/CEBaB-v1.1/dev.json")
elif data_type == "aug_cebab":
    num_concept_labels = 10
    train_split = "train_aug_cebab"
    test_split = "test_aug_cebab"
    val_split = "val_aug_cebab"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../dataset/cebab/train_cebab_new_concept_single.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/cebab/test_cebab_new_concept_single.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/cebab/dev_cebab_new_concept_single.csv")
elif data_type == "aug_yelp":
    num_concept_labels = 10
    train_split = "train_aug_yelp"
    test_split = "test_aug_yelp"
    val_split = "val_aug_yelp"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../dataset/cebab/train_yelp_exclusive_new_concept_single.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/cebab/test_yelp_new_concept_single.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/cebab/dev_yelp_new_concept_single.csv")
elif data_type == "aug_cebab_yelp":
    num_concept_labels = 10
    train_split = "train_aug_cebab_yelp"
    test_split = "test_aug_cebab_yelp"
    val_split = "val_aug_cebab_yelp"
    train_split_cebab = pd.read_csv("../dataset/cebab/train_cebab_new_concept_single.csv")
    test_split_cebab = pd.read_csv("../dataset/cebab/test_cebab_new_concept_single.csv")
    val_split_cebab = pd.read_csv("../dataset/cebab/dev_cebab_new_concept_single.csv")
    train_split_yelp = pd.read_csv("../dataset/cebab/train_yelp_exclusive_new_concept_single.csv")
    test_split_yelp = pd.read_csv("../dataset/cebab/test_yelp_new_concept_single.csv")
    val_split_yelp = pd.read_csv("../dataset/cebab/dev_yelp_new_concept_single.csv")
    CEBaB = {}
    CEBaB[train_split] = pd.concat([train_split_cebab, train_split_yelp], ignore_index=True)
    CEBaB[test_split] = pd.concat([test_split_cebab, test_split_yelp], ignore_index=True)
    CEBaB[val_split] = pd.concat([val_split_cebab, val_split_yelp], ignore_index=True)

print("Finish loading data {}".format(data_type))
# Define a custom dataset class for loading the data

class MyDataset(Dataset):
    # Split = train/dev/test
    def __init__(self, split, skip_class = "no majority"):
        self.data = CEBaB[split]
        self.labels = self.data["review_majority"]
        self.text = self.data["description"]
       
        self.food_aspect = self.data["food_aspect_majority"]
        self.ambiance_aspect = self.data["ambiance_aspect_majority"]
        self.service_aspect = self.data["service_aspect_majority"]
        self.noise_aspect =self.data["noise_aspect_majority"]

        if data_type != "pure_cebab":
            # cleanliness price	location	menu variety	waiting time	waiting area	## parking	wi-fi	kids-friendly
            self.cleanliness_aspect = self.data["cleanliness"]
            self.price_aspect = self.data["price"]
            self.location_aspect = self.data["location"]
            self.menu_variety_aspect = self.data["menu variety"]
            self.waiting_time_aspect =self.data["waiting time"]
            self.waiting_area_aspect =self.data["waiting area"]

        self.map_dict = {"Negative":0, "Positive":1, "unknown":2, "":2,"no majority":2}

        self.skip_class = skip_class
        if skip_class is not None:
            self.indices = [i for i, label in enumerate(self.labels) if label != skip_class]
        else:
            self.indices = range(len(self.labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        text = self.text[self.indices[index]]
        label = int(self.labels[self.indices[index]]) - 1

        # gold labels
        food_concept = self.map_dict[self.food_aspect[self.indices[index]]]
        ambiance_concept = self.map_dict[self.ambiance_aspect[self.indices[index]]]
        service_concept = self.map_dict[self.service_aspect[self.indices[index]]]
        noise_concept = self.map_dict[self.noise_aspect[self.indices[index]]]
        
        if data_type != "pure_cebab":
            # noisy labels
            #cleanliness price	location	menu variety	waiting time	waiting area	## parking	wi-fi	kids-friendly
            cleanliness_concept = self.map_dict[self.cleanliness_aspect[self.indices[index]]]
            price_concept = self.map_dict[self.price_aspect[self.indices[index]]]
            location_concept = self.map_dict[self.location_aspect[self.indices[index]]]
            menu_variety_concept = self.map_dict[self.menu_variety_aspect[self.indices[index]]]
            waiting_time_concept = self.map_dict[self.waiting_time_aspect[self.indices[index]]]
            waiting_area_concept = self.map_dict[self.waiting_area_aspect[self.indices[index]]]

        if data_type != "pure_cebab":
            concept_labels = [food_concept,ambiance_concept,service_concept,noise_concept,cleanliness_concept,price_concept,location_concept,menu_variety_concept,waiting_time_concept,waiting_area_concept]
        else: 
            concept_labels = [food_concept,ambiance_concept,service_concept,noise_concept]

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        if data_type != "pure_cebab":
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
                "food_concept": torch.tensor(food_concept, dtype=torch.long),
                "ambiance_concept": torch.tensor(ambiance_concept, dtype=torch.long),
                "service_concept": torch.tensor(service_concept, dtype=torch.long),
                "noise_concept": torch.tensor(noise_concept, dtype=torch.long),
                "cleanliness_concept": torch.tensor(cleanliness_concept, dtype=torch.long),
                "price_concept": torch.tensor(price_concept, dtype=torch.long),
                "location_concept": torch.tensor(location_concept, dtype=torch.long),
                "menu_variety_concept": torch.tensor(menu_variety_concept, dtype=torch.long),
                "waiting_time_concept": torch.tensor(waiting_time_concept, dtype=torch.long),
                "waiting_area_concept": torch.tensor(waiting_area_concept, dtype=torch.long),
                "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
            }
        else:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
                "food_concept": torch.tensor(food_concept, dtype=torch.long),
                "ambiance_concept": torch.tensor(ambiance_concept, dtype=torch.long),
                "service_concept": torch.tensor(service_concept, dtype=torch.long),
                "noise_concept": torch.tensor(noise_concept, dtype=torch.long),
                "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
            }


# Load the data
train_dataset = MyDataset(train_split)
test_dataset = MyDataset(test_split)
val_dataset = MyDataset(val_split)

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


#Set ModelXtoCtoY_layer
# concept_classes 每个concept有几类；    label_classes  label的个数；  n_attributes concept的个数； n_class_attr 每个concept有几类；
ModelXtoCtoY_layer = ModelXtoCtoY_function(concept_classes = num_each_concept_classes, label_classes = num_labels, n_attributes = num_concept_labels, bottleneck = True, expand_dim = 0, n_class_attr=num_each_concept_classes, use_relu=False, use_sigmoid=False, model_type=model_type, aux_logits=is_aux_logits)


# Set up the optimizer and loss function
# optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)
optimizer = torch.optim.Adam(list(model.parameters()) + list(ModelXtoCtoY_layer.parameters()), lr=3e-4)


if model_name == 'lstm':
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# classifier.to(device)


# Modify model with sparse mask
ModelXtoCtoY_layer.to(device)
model = insert_sparse_mask(model, num_concept_labels, None, pruning_type='obert')
model.to(device)


FINETUNE_EPOCH=args.FINETUNE_EPOCH
START_EPOCH=args.START_EPOCH
END_EPOCH=args.END_EPOCH
INIT_SPARSITY=args.INIT_SPARSITY
FINAL_SPARSITY=args.FINAL_SPARSITY

M_GRAD=args.M_GRAD
BLOCK_SIZE=args.BLOCK_SIZE
LAMBD=args.LAMBD
NUM_LAYERS = args.NUM_LAYERS
EPS=torch.finfo(torch.float32).eps

#step 1.1  XtoCtoY finetune with fixed mask
print("train XtoCtoY! Fix Sparse Mask")
for epoch in range(FINETUNE_EPOCH):
    predicted_concepts_train = []
    predicted_concepts_train_label = []
    ModelXtoCtoY_layer.train()
    model.train()
    
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        food_concept = batch["food_concept"].to(device)
        ambiance_concept=batch["ambiance_concept"].to(device)
        service_concept=batch["service_concept"].to(device)
        noise_concept=batch["noise_concept"].to(device)

        if data_type != "pure_cebab":
            cleanliness_concept = batch["cleanliness_concept"].to(device)
            price_concept = batch["price_concept"].to(device)
            location_concept = batch["location_concept"].to(device)
            menu_variety_concept = batch["menu_variety_concept"].to(device)
            waiting_time_concept = batch["waiting_time_concept"].to(device)
            waiting_area_concept = batch["waiting_area_concept"].to(device)                
        concept_labels=batch["concept_labels"].to(device)
        concept_labels = torch.t(concept_labels)
        concept_labels = concept_labels.contiguous().view(-1) 

        optimizer.zero_grad()

        XtoC_outputs = []
        XtoY_outputs = []
        for concept_idx in range(num_concept_labels):
            set_sparse_index(model, concept_idx)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if model_name == 'lstm':
                pooled_output = outputs
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  

            outputs  = ModelXtoCtoY_layer(pooled_output) 

            XtoC_outputs.append(outputs[concept_idx+1]) 
            XtoY_outputs.extend(outputs[0:1])

        # XtoC_loss
        XtoC_logits = torch.nn.Sigmoid()(torch.cat(XtoC_outputs, dim=0)) # 32*4 00000000111111112222222233333333
        XtoC_loss = loss_fn(XtoC_logits, concept_labels)
        # XtoY_loss
        # print(len(XtoY_outputs))
        # Y_batch = torch.stack(XtoY_outputs).mean(0)
        Y_batch = XtoY_outputs[0]
        XtoY_loss = loss_fn(Y_batch, label)
        loss = XtoC_loss*lambda_XtoC+XtoY_loss
        loss.backward()
        optimizer.step()

    model.eval()
    ModelXtoCtoY_layer.eval()
    val_accuracy = 0.
    concept_val_accuracy = 0.
    test_accuracy = 0.
    concept_test_accuracy = 0.
    best_acc_score = 0
    predict_labels = np.array([])
    true_labels = np.array([])
    concept_predict_labels = np.array([])
    concept_true_labels = np.array([])
    predict_concepts = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            food_concept = batch["food_concept"].to(device)
            ambiance_concept=batch["ambiance_concept"].to(device)
            service_concept=batch["service_concept"].to(device)
            noise_concept=batch["noise_concept"].to(device)
            if data_type != "pure_cebab":
                cleanliness_concept = batch["cleanliness_concept"].to(device)
                price_concept = batch["price_concept"].to(device)
                location_concept = batch["location_concept"].to(device)
                menu_variety_concept = batch["menu_variety_concept"].to(device)
                waiting_time_concept = batch["waiting_time_concept"].to(device)
                waiting_area_concept = batch["waiting_area_concept"].to(device)        
            concept_labels=batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels)
            concept_labels = concept_labels.contiguous().view(-1)


            XtoC_outputs = []
            XtoY_outputs = []
            for concept_idx in range(num_concept_labels):
                set_sparse_index(model, concept_idx)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if model_name == 'lstm':
                    pooled_output = outputs
                else:
                    pooled_output = outputs.last_hidden_state.mean(1)  

                outputs  = ModelXtoCtoY_layer(pooled_output)  
                XtoC_outputs.append(outputs[concept_idx+1]) 
                XtoY_outputs.extend(outputs [0:1])

            
            predictions = torch.argmax(XtoY_outputs[0], axis=1)
            # predictions = torch.argmax(torch.stack(XtoY_outputs).mean(0), axis=1)
            val_accuracy += torch.sum(predictions == label).item()
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
            #concept accuracy
            XtoC_logits = torch.cat(XtoC_outputs, dim=0)
            concept_predictions = torch.argmax(XtoC_logits, axis=1)
            concept_val_accuracy += torch.sum(concept_predictions == concept_labels).item()
            concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())
            concept_predictions = concept_predictions.reshape(-1,num_concept_labels)  # reshape 二维向量[batch_size*num_concept_labels]
        
        val_accuracy /= len(val_dataset)
        num_labels = len(np.unique(true_labels))

        concept_val_accuracy /= len(val_dataset)
        concept_num_true_labels = len(np.unique(concept_true_labels))
        
        macro_f1_scores = []
        for label in range(num_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

        concept_macro_f1_scores = []
        for concept_label in range(concept_num_true_labels):
            concept_label_pred = np.array(concept_predict_labels) == concept_label
            concept_label_true = np.array(concept_true_labels) == concept_label
            concept_macro_f1_scores.append(f1_score(concept_label_true, concept_label_pred, average='macro'))
            concept_mean_macro_f1_score = np.mean(concept_macro_f1_scores)

    print(f"Epoch {epoch + 1}: Val concept Acc = {concept_val_accuracy*100/num_concept_labels} Val concept Macro F1 = {concept_mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Val Acc = {val_accuracy*100} Val Macro F1 = {mean_macro_f1_score*100}")
    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        torch.save(model, model_path+model_name+"_"+str(FINAL_SPARSITY)+"_joint.pth")
        torch.save(ModelXtoCtoY_layer, model_path+model_name+"_"+str(FINAL_SPARSITY)+"_ModelXtoCtoY_layer_joint.pth")


    #step 1.2  Update Mask
    if epoch >= START_EPOCH and epoch <= END_EPOCH:
        target_sparsity = (FINAL_SPARSITY - INIT_SPARSITY) * (epoch - START_EPOCH) / (END_EPOCH - START_EPOCH) + INIT_SPARSITY
        print('Start Pruning for Target-Sparsity {}, density'.format(target_sparsity, 1 - target_sparsity))

        for concept_idx in range(num_concept_labels):
            print('Concept: {}'.format(concept_idx+1))
            set_sparse_index(model, concept_idx)
            grad_steps = 0

            all_weight_name_list = []
            all_weight_module_list = {}
            update_weight_name_list = []
            for name, module in model.named_modules():
                if isinstance(module, SubnetLinear_Mask):
                    all_weight_name_list.append(name)
                    all_weight_module_list[name] = module

            num_of_weights = len(all_weight_name_list)
            iterative_steps = int(num_of_weights / NUM_LAYERS)

            for update_steps in range(iterative_steps+1):
                update_weight_list = []
                for name_index in range(update_steps*NUM_LAYERS, min((update_steps+1)*NUM_LAYERS, num_of_weights)):
                    update_weight_list.append(all_weight_name_list[name_index])

            finnvs_dict = {}
            for name in update_weight_list:
                finnvs_dict[name] = EmpiricalBlockFisherInverse(
                    num_grads = M_GRAD,
                    fisher_block_size = BLOCK_SIZE,
                    num_weights = all_weight_module_list[name].mask_list[concept_idx].numel(),
                    damp = LAMBD,
                    device = device,
                )
            print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

            for batch in tqdm(train_loader, desc="Training", unit="batch"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
                food_concept = batch["food_concept"].to(device)
                ambiance_concept=batch["ambiance_concept"].to(device)
                service_concept=batch["service_concept"].to(device)
                noise_concept=batch["noise_concept"].to(device)
                if data_type != "pure_cebab":
                    cleanliness_concept = batch["cleanliness_concept"].to(device)
                    price_concept = batch["price_concept"].to(device)
                    location_concept = batch["location_concept"].to(device)
                    menu_variety_concept = batch["menu_variety_concept"].to(device)
                    waiting_time_concept = batch["waiting_time_concept"].to(device)
                    waiting_area_concept = batch["waiting_area_concept"].to(device)                
                concept_labels=batch["concept_labels"].to(device)
                concept_labels = torch.t(concept_labels)
                concept_labels = concept_labels.contiguous().view(-1) 

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state.mean(1)
                outputs  = ModelXtoCtoY_layer(pooled_output)
                XtoC_output = outputs[concept_idx+1]
                XtoC_logits = torch.nn.Sigmoid()(XtoC_output)
                XtoY_output = outputs[0:1]

                BS=XtoC_logits.shape[0]

                XtoC_loss = loss_fn(XtoC_logits, concept_labels[concept_idx*BS: concept_idx*BS+BS])
                XtoY_loss = loss_fn(XtoY_output[0], label)

                loss = XtoC_loss*lambda_XtoC+XtoY_loss
                loss.backward()

                for name in update_weight_list:
                    if not name in finnvs_dict: continue
                    if all_weight_module_list[name].fc.weight.grad == None:
                        del finnvs_dict[name]
                        continue

                    finnvs_dict[name].add_grad(all_weight_module_list[name].fc.weight.grad.reshape(-1))

                grad_steps += 1
                
                if grad_steps >= M_GRAD:
                    break


            # Calculate Scores and Update Mask
            for name, module in model.named_modules():
                if isinstance(module, SubnetLinear_Mask):

                    if not name in finnvs_dict: continue
                    scores = (
                        (module.fc.weight.data.reshape(-1) ** 2).to(device)
                        / (2.0 * finnvs_dict[name].diag() + EPS)
                    ).reshape(module.fc.weight.shape)
                    d = module.mask_list[concept_idx].numel()
                    kth_score = torch.kthvalue(scores.reshape(-1), round(target_sparsity * d))[0]
                    module.mask_list[concept_idx].data = (scores > kth_score).to(module.mask_list[concept_idx].dtype)
                    print('Remaining weight for {} = {:.4f}'.format(name, module.mask_list[concept_idx].gt(0).float().mean()))

####################### test
num_epochs = 1
print("Test!")
model = torch.load(model_path+model_name+"_"+str(FINAL_SPARSITY)+"_joint.pth")
ModelXtoCtoY_layer = torch.load(model_path+model_name+"_"+str(FINAL_SPARSITY)+"_ModelXtoCtoY_layer_joint.pth") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            food_concept = batch["food_concept"].to(device)
            ambiance_concept=batch["ambiance_concept"].to(device)
            service_concept=batch["service_concept"].to(device)
            noise_concept=batch["noise_concept"].to(device)
            if data_type != "pure_cebab":
                cleanliness_concept = batch["cleanliness_concept"].to(device)
                price_concept = batch["price_concept"].to(device)
                location_concept = batch["location_concept"].to(device)
                menu_variety_concept = batch["menu_variety_concept"].to(device)
                waiting_time_concept = batch["waiting_time_concept"].to(device)
                waiting_area_concept = batch["waiting_area_concept"].to(device)        
            concept_labels=batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels)
            concept_labels = concept_labels.contiguous().view(-1)

            XtoC_outputs = []
            XtoY_outputs = []
            for concept_idx in range(num_concept_labels):
                set_sparse_index(model, concept_idx)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if model_name == 'lstm':
                    pooled_output = outputs
                else:
                    pooled_output = outputs.last_hidden_state.mean(1)  

                outputs  = ModelXtoCtoY_layer(pooled_output)  
                XtoC_outputs.append(outputs[concept_idx+1]) 
                XtoY_outputs.extend(outputs [0:1])

            predictions = torch.argmax(XtoY_outputs[0], axis=1)
            # predictions = torch.argmax(torch.stack(XtoY_outputs).mean(0), axis=1)
            test_accuracy += torch.sum(predictions == label).item()
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
            #concept accuracy
            XtoC_logits = torch.cat(XtoC_outputs, dim=0)
            concept_predictions = torch.argmax(XtoC_logits, axis=1)
            concept_test_accuracy += torch.sum(concept_predictions == concept_labels).item()
            concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())
            concept_predictions = concept_predictions.reshape(-1,num_concept_labels)  # reshape 二维向量[batch_size*num_concept_labels]
        
        test_accuracy /= len(test_dataset)
        num_labels = len(np.unique(true_labels))

        concept_test_accuracy /= len(test_dataset)
        concept_num_true_labels = len(np.unique(concept_true_labels))
        
        macro_f1_scores = []
        for label in range(num_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

        concept_macro_f1_scores = []
        for concept_label in range(concept_num_true_labels):
            concept_label_pred = np.array(concept_predict_labels) == concept_label
            concept_label_true = np.array(concept_true_labels) == concept_label
            concept_macro_f1_scores.append(f1_score(concept_label_true, concept_label_pred, average='macro'))
            concept_mean_macro_f1_score = np.mean(concept_macro_f1_scores)

    print(f"Epoch {epoch + 1}: Test concept Acc = {concept_test_accuracy*100/num_concept_labels} Test concept Macro F1 = {concept_mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Test Acc = {test_accuracy*100} Test Macro F1 = {mean_macro_f1_score*100}")
