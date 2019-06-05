import json

def load_json(filename):
    with open(filename,'r') as f:
        return json.load(f)

if __name__ == '__main__':
    datalist = load_json('phase_liverfibrosis.json')
    print(datalist)