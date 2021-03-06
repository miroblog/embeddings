
import os
import re
import pickle
import pandas as pd

def parse_file_name(file_name):
    tokens = re.split("[_,]", file_name)
    param = {}
    for i, token in enumerate(tokens):
        token = token.strip()
        if (i == 0):
            param['type'] = token
        if (i == 1):
            param['model'] = token
        if("=" in token):
            name, value = token.split("=")
            param[name] = value
    return param

def get_final_result(file):
    history = pickle.load(open(PATH+"/"+file, "rb"))
    result = {}
    result['val_loss'] = max(history['val_loss'])
    result['val_acc'] = max(history['val_acc'])
    result['acc'] = max(history['acc'])
    result['loss'] = max(history['loss'])
    return result

def main(path):
    files = os.listdir(path)
    df = pd.DataFrame()
    for filename in files:
        param = parse_file_name(filename)
        result = get_final_result(filename)
        param['val_loss'] = result['val_loss']
        param['val_acc'] = result['val_acc']
        param['acc'] = result['acc']
        param['loss'] = result['loss']
        df = df.append({ 'type' : param['type'],
                         'model': param['model'],
                         'window':param['window'],
                        'size':param['size'],
                        'count':param['count'],
                         'acc' : param['acc'],
                         'loss' : param['loss'],
                        'val_loss':param['val_loss'],
                        'val_acc':param['val_acc']}, ignore_index=True)
    df = df[['type', 'model', 'window', 'size', 'count', 'acc', 'loss', 'val_acc', 'val_loss']]
    df = df.drop_duplicates(subset=['type', 'model', 'window', 'size', 'count'],
                            keep='first',inplace=False)
    df.to_csv(args.out_name+".csv")
    print("check for:",args.out_name+".csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="./history", help="specify history directory")
    parser.add_argument("-o", "--out_name", default="history_result", help="output file name")
    args = parser.parse_args()
    PATH = args.path
    main(path = PATH)

