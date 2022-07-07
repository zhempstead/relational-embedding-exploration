import pickle


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
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    def load(self, input_path):
        with open(input_path, "rb") as f:
            tmp = pickle.load(f)
        self.cnt = tmp.cnt
        self.token2id = tmp.token2id
        self.id2token = tmp.id2token

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
            import pdb; pdb.set_trace()
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
