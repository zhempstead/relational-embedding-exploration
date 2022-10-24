import pandas as pd

class TokenDict:
    def __init__(self, path=None):
        # token -> numerical id:
        # string -> string
        if path is None:
            self.token2id = dict()
            self.id2token = dict()
            self.cnt = 0
        else:
            self.load(path)

    def save(self, output_path):
        df = pd.DataFrame({"id": self.token2id.values(), "token": self.token2id.keys()})
        df.to_feather(output_path)

    def load(self, input_path):
        df = pd.read_feather(input_path)
        self.cnt = len(df)
        self.token2id = dict(zip(df['token'], df['id']))
        self.id2token = dict(zip(df['id'], df['token']))

    def display(self):
        print("cnt", self.cnt)
        print("token2id", self.token2id)

    def put(self, token):
        token = str(token)
        if token not in self.token2id.keys():
            self.token2id[token] = str(self.cnt)
            self.id2token[str(self.cnt)] = token
            self.cnt += 1
        return self.token2id[token]

    def getNumForToken(self, token):
        token = str(token)
        if token not in self.token2id.keys():
            return None
        else:
            return self.token2id[token]

    def getTokenForNum(self, num):
        num = str(num)
        if num not in self.id2token.keys():
            return None
        else:
            return self.id2token[num]

    def getTokenForNums(self, lst):
        return [self.getTokenForNum(x) for x in lst]

    def getAllTokensWith(self, token):
        lst = []
        token = str(token)
        lst = [key for key in self.token2id.keys() if token in key]
        return lst
