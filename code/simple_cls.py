import torch
import os
import dill 
from pathlib import Path
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator, Dataset, Pipeline, RawField, LabelField
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
import torch.optim as optim
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, mean_squared_error
import logging
import re
import sys
import json
import wandb
import torch.utils.checkpoint as cp

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


## Save datasets
def save_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, path.parent / (path.name +".examples.pkl"), pickle_module=dill)
    torch.save(dataset.fields, path.parent / (path.name+".fields.pkl"), pickle_module=dill)

def load_dataset(path):
    if not isinstance(path, Path):
        path = Path(path)
    examples = torch.load(path.parent / (path.name+".examples.pkl"), pickle_module=dill)
    fields = torch.load(path.parent / (path.name+".fields.pkl"), pickle_module=dill)
    return Dataset(examples, fields)

def check_datasets(path):
    if not isinstance(path, Path):
        path = Path(path)
    #if os.path.isfile( os.path.join(path, "examples.pkl")) and os.path.isfile(os.path.join(path, "fields.pkl")):
    if os.path.isfile(path.parent / (path.name+".examples.pkl")) and os.path.isfile(path.parent / (path.name+".fields.pkl")):
        return True
    else:
        return False


class DummyModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, text, dummy_tensor=None):
        ret = self.module(text)
        return ret[0]

class BERT(nn.Module):

    def __init__(self, base='bert', options_name="bert-base-uncased", num_labels=1, loss = None, multilabel=False, class_weights=None, checkpointing=False ):
        super(BERT, self).__init__()        
        if base == 'bert':
            self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels= num_labels )
        elif base == 'gpt2':
            self.encoder = GPT2ForSequenceClassification.from_pretrained(options_name, num_labels= num_labels )
            if self.encoder.config.pad_token_id is None:#self.encoder.tokenizer.pad_token is None:
                self.encoder.config.pad_token_id = self.encoder.config.eos_token_id 
        else:
            self.encoder = RobertaForSequenceClassification.from_pretrained(options_name, num_labels = num_labels )
        self.loss_fnc = loss
        self.num_labels = num_labels
        self.multilabel = multilabel
        self.class_weights = class_weights
        self.checkpointing = checkpointing
        self.module_wrapper = DummyModule(self.encoder)
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
 
    #def forward(self, text, batchsize= 1, labels=None, training=False):
    #    if self.checkpointing and training:
    #        loss,logits = cp.checkpoint( self.__forward__, text, batchsize, labels )
    #        print("loss", loss)
    #        print("logits", logits)
    #        return loss, logits
    #    else:
    #        return self.__forward__( text, batchsize=batchsize, labels=labels ) 
        
    def forward(self, text, batchsize=1, labels=None, training=False):
        #logits = self.encoder(text)[0]
        #logits = self.module_wrapper(text)[0]
        if self.checkpointing and training:
            logits = cp.checkpoint( self.module_wrapper, text, self.dummy_tensor )
        else:
            logits = self.module_wrapper( text, self.dummy_tensor )

        #regression; use MAE loss since we're mostly doing reddit stuff 
        if self.num_labels == 1:
            labels = labels.float()
            #loss_fct = self.loss_fnc if self.loss_fnc else nn.L1Loss() 
            loss_fct = self.loss_fnc if self.loss_fnc else nn.MSELoss()
            loss = loss_fct( logits.view(-1), labels.view(-1) )
        #textcat
        else:
            if self.multilabel:
                labels = labels.unsqueeze(1)
                #print(labels)
                #loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)#self.loss_fnc if self.loss_fnc else nn.BCELoss()
                loss_fct = nn.MultiLabelSoftMarginLoss(weight=self.class_weights)
                #print( type(logits.view( -1, self.num_labels )[0]) )
                #print( type( labels.view(1, self.num_labels) [0]))
                #loss = loss_fct(logits.view( -1, self.num_labels ), labels.view(batchsize, self.num_labels).float() )
                loss = loss_fct(logits.view( -1, self.num_labels ), labels.view(batchsize, self.num_labels).long(), )
            else:
                loss_fct = self.loss_fnc if self.loss_fnc else nn.CrossEntropyLoss()
                loss = loss_fct(logits.view( -1, self.num_labels ), labels.view(-1) )
    
        return loss, logits


# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss, valid_score):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss,
                  'valid_score': valid_score}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, args):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=args.device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']




def save_metrics(save_path, metrics_dict, fmt='pt'): 
    if save_path == None:
        return
    
    if fmt == 'json':
        with open(save_path, 'w+') as f:
            json.dump( metrics_dict, f )
    else:
        torch.save(metrics_dict, save_path)
    print(f'Metrics saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']



# Training Function

def train(model,
          optimizer,
          scheduler, 
          train_loader,
          valid_loader, 
          args, 
          loss_fct = None,
          num_epochs = 5,
          eval_every = None,
          best_valid_loss = float("Inf")):
    
    file_path = args.dest
    device = args.device
    # initialize running values
    running_loss = 0.0
    global_step = 0
    if eval_every is None:
        eval_every = len(train_loader) // 2

    train_loss_list = []
    valid_loss_list = []
    valid_score_list = []
    global_steps_list = []
    model.zero_grad()
    if args.wandb:
        wandb.init(project='temporal_drift.'+args.wandbname)
        wandb.watch(model)

    # training loop
    for epoch in range(num_epochs):
        #for text,labels in tqdm(train_loader,desc='training' ):

        for batch in tqdm(train_loader,desc='training' ):
            batchsize = batch.batch_size
            labels = batch.score
            text = batch.text
            #print(labels.shape)
            #print(labels)
            #print(text.shape)
            #print(text)
            model.train()

            #if model.multilabel:
            #    labels = labels.cpu().numpy().astype(int)
            #    targets = np.zeros( model.num_labels )
            #    targets[ labels ] = 1
            #    labels = torch.from_numpy( targets ) 
            labels = labels.type(torch.LongTensor)  
            labels = labels.to(device)
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            loss, logits = model(text, labels=labels, batchsize=batchsize, training=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                valid_running_loss, valid_score = evaluate(model, valid_loader, args)
               
                # evaluation
                average_train_loss = running_loss / global_step
                average_valid_loss = valid_running_loss# / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                valid_score_list.append( valid_score['accuracy'] ) 
                global_steps_list.append(global_step)
            

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Score:{:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss, valid_score['accuracy']))
                if args.wandb:
                    wandb.log({"epoch": epoch+1, "step": global_step, "average_train_loss": average_train_loss, "average_valid_loss": average_valid_loss, "valid_score": valid_score['accuracy']})


                metrics_dict = {'train_loss_list': train_loss_list,
                    'valid_loss_list': valid_loss_list,
                    'global_steps_list': global_steps_list,
                    'valid_score_list': valid_score_list,
                    }
 


                # checkpoint
                average_valid_loss = -valid_score['f1']
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path  + 'model.pt', model, best_valid_loss, valid_score)
                    save_metrics(file_path  + 'metrics.pt', metrics_dict)

    
    #save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')



def evaluate(model, valid_loader, args, loss_fct = None):
    valid_running_loss = 0
    eval_steps = 0  
    preds = None
    out_labels = None
    device = args.device
    #results = {}

    for batch in tqdm(valid_loader, desc='evaluation'):
        model.eval()
        with torch.no_grad():
            #if model.multilabel:
            #    labels = labels.cpu().numpy().astype(int)
            #    targets = np.zeros( model.num_labels )
            #    targets[ labels ] = 1
            #    labels = torch.from_numpy( targets ) 
            batchsize = batch.batch_size
            labels = batch.score
            text = batch.text
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            if model.num_labels == 1:
                labels = labels.float()
            output = model(text, labels=labels, batchsize=batchsize)
            loss, logits = output[:2]                    
            valid_running_loss += loss.item()
        
        eval_steps += 1  

        if model.multilabel:
            if preds is None:
                preds = logits.detach().cpu().numpy()
                #out_labels = []
                #out_labels.append( labels.detach().cpu().numpy() )
                out_labels = labels.detach().cpu().numpy()

                #out_labels= np.append( out_labels, labels.detach().cpu().numpy() )
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                #out_labels.append( labels.detach().cpu().numpy())
                out_labels = np.concatenate( (out_labels, labels.detach().cpu().numpy()) )

        else:         
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_labels = np.append( out_labels, labels.detach().cpu().numpy(), axis=0)
    valid_running_loss = valid_running_loss / eval_steps
    #its actually the same as loss with regression 
    if model.num_labels == 1:
        print("Regression")
        preds = np.squeeze(preds)
        #print( "pred", type(preds), preds.shape )
        #print( "out_labels", type(out_labels), out_labels.shape)
        #loss_fn = nn.L1Loss()
        #score = loss_fn( out_labels, preds )
        #score = -1 * mean_absolute_error( out_labels, preds )
        score = {'accuracy':-1 * mean_squared_error( out_labels, preds )}
 
    else:
        if model.multilabel:
            print('multilabel')
            threshold = .0
            out_labels = np.array( out_labels ) 
            #print( "labels", preds )
            print( "labels", preds ) 
            preds = np.array( preds > threshold, dtype = float )
            #print( "labels", preds )
            distr = [ 0.0 for i in range( model.num_labels )]
                        
            #print( out_labels.shape, preds.shape )
            nz1 =  np.nonzero(preds)[1]
            for idx in nz1:
                ind = int( idx % model.num_labels )
                distr[ind] += 1 / out_labels.shape[0]
            nz = np.nonzero(preds)[1] % model.num_labels 
            
            uniqs = np.unique(nz)
            print( len(uniqs), uniqs)

            score = {'accuracy':  accuracy_score( out_labels, preds ),
                    'micro_f1' : f1_score( out_labels, preds, average='micro'),# labels=uniqs  ),
                    'macro_f1' : f1_score( out_labels, preds, average  = 'macro'),#  labels=uniqs ),
                    'weighted_f1': f1_score( out_labels, preds, average  = 'weighted'),#  labels=uniqs ),
                    'distribution': distr,
                    }
        else:
            print( 'classification' )
            print( 'nonarg', preds )
            preds = np.argmax(preds, axis=1)
            #score = accuracy_score( out_labels, preds) 
            #print( 'preds', preds )
            score = {'accuracy': accuracy_score( out_labels, preds ),
                    'f1': f1_score( out_labels, preds, average='macro' ) }
        
    return valid_running_loss, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=False)
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--num_labels', type=int, default=1,
                        help='number of classes; n=1 for regression and n>1 for textcat')   
 
    parser.add_argument('--batchsize', type=int, default = 32 )

    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--num_epoch", default=20, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--chkpt", default=None, type=str)
    parser.add_argument("--multilabel", action="store_true") 
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--wandb", default=None, type=str)
    parser.add_argument("--wandbname", default=None, type=str)
    parser.add_argument("--yelp", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()

    if args.wandb:
        os.environ["WANDB_API_KEY"] = args.wandb

    ## GPU Setup ##
    if args.gpu ==-1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        print("HELLO?")
        if args.gpu:
            device = torch.device("cuda:"+str(args.gpu))
        else:
            device = torch.device("cuda")
        args.n_gpu = 1
    args.device = device 

    ## Tokenizer Setup ##
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name)#'bert-base-uncased')
        tokenizer_name = 'bert'

    elif args.model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer_name='gpt2'
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer_name = 'roberta'

    MAX_SEQ_LEN = 256
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


    ## Dataset ##
    label = LabelField(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float, is_target=True)
    text = Field(
        use_vocab=False, tokenize= lambda a: tokenizer.encode(a,  max_length=512), 
        lower=False, include_lengths=False, batch_first=True,
        fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX,
        )
    


    if args.num_labels == 1 or  args.yelp:
        fields = {
            'score': ('score', label), 
            'text': ('text', text )
            }
    else:
        if args.multilabel:
            def convert_to_onehot(lbl):
                labels = lbl
                #labels = np.array( lbl ) 
                targets = np.zeros( args.num_labels )
                targets[labels] = 1
                return targets

            convert_to_one_hot_pipe  = Pipeline( lambda a,b: convert_to_onehot(a) ) 
            #convert_to_one_hot_pipe  = Pipeline( lambda a,b: [convert_to_onehot(y) for y in a ] )

            label = LabelField( sequential=False, use_vocab=False, batch_first=True, dtype=torch.float, postprocessing=convert_to_one_hot_pipe)
            fields = {
                'labels': ('score', label), 
                'text': ('text', text )
                }
        else:
            fields = {
                'label': ('score', label),
                'text': ('text', text )
                } 


    cache_fn = lambda a: os.path.join(args.cache_dir if args.cache_dir else args.dest, '{}.{}.cached'.format(os.path.basename(a), tokenizer_name))
    print(cache_fn(args.train_file ))
    if check_datasets(cache_fn(args.train_file)) and check_datasets(cache_fn(args.eval_file)):
        #trn = load_dataset( cache_fn(args.train_file ))
        #dev = load_dataset( cache_fn(args.eval_file))
        pass 
    else:
        #train, dev, test = TabularDataset.splits(path=" ",  train=args.train_file, validation=args.eval_file, test =args.test_file,
        #                                    format='json', fields=fields,
        #                                )

        trn = TabularDataset(args.train_file, format='json', fields=fields)
        dev   = TabularDataset(args.eval_file, format='json', fields=fields)
        #test  = TabularDataset(args.test_file, format='json', fields=fields)        
        
        print(type(trn)) 
        #save_dataset( trn, cache_fn(args.train_file)  )
        #save_dataset( dev,   cache_fn(args.eval_file) ) 
        #save_dataset( test,  cache_fn(args.test_file) )


    # Iterators
    train_iter = BucketIterator(trn, batch_size=args.batchsize,shuffle=True,
                                device=device, train=True, sort=False)
    valid_iter = BucketIterator(dev, batch_size=1, 
                                device=device, train=False, sort=False)
    #test_iter = Iterator(test, batch_size=args.batchsize, device=device, train=False, shuffle=False, sort=False)
    dummy_iter = Iterator( trn, batch_size=1, train=False, sort=False, shuffle=False) 
    
    class_weights = torch.from_numpy(np.array( [1.0 for i in range( args.num_labels )] ))
    total = len( dummy_iter )
    for item in dummy_iter:
        class_weights = class_weights + item.score 
    class_weights = class_weights / total 
    class_weights = torch.reciprocal( class_weights )
    if args.multilabel: 
    	class_weights = class_weights.to(device) 

    model = None
    model = BERT(base=args.model_type, options_name = args.model_name, num_labels=args.num_labels, \
        multilabel=args.multilabel, class_weights=class_weights,checkpointing=args.checkpointing  ).to(device)
    if args.chkpt:
        load_checkpoint(args.chkpt, model, args)
    if args.do_train:
        if args.max_steps > 0:
            t_total = args.max_steps
            num_epoch = args.max_steps // (len(trn)) + 1
        else:
            t_total = len(trn) * args.num_epoch
            num_epoch = args.num_epoch 
        #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        #optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        #if hasattr(torch.cuda, 'empty_cache'):
        #torch.cuda.empty_cache()
        train(model, optimizer, scheduler, train_iter, valid_iter, args, num_epochs = num_epoch)
    if args.do_eval:
        loss, score = evaluate(model, valid_iter, args )
        eval_base = os.path.splitext(os.path.basename(args.eval_file))[0]
        eval_base = eval_base[: eval_base.index('.')]
        print( "Validation Loss :: ", loss )
        print( "Validation Score :: ", score )
        metrics_dict = { "Valid Loss": loss, "valid score": score, "command":sys.argv }
        save_metrics( args.dest + eval_base + '.eval_scores.json', metrics_dict, fmt='json' )
        
set_global_logging_level(logging.ERROR)
main()
