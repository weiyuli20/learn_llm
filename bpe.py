from collections import Counter,deque
from functools import lru_cache
import json

class BPETokenizerSimple:
    def __init__(self):
        # maps token_id to token_str {1:"some"}
        self.vocab = {}
        # maps token_str to token_id {"some":1}
        self.inverse_vocab = {}
        #Dictonary of bpe_merges:
        self.bpe_merges = {}
    

    def train(self,text,vocab_size,allowed_special ={'<|endoftext|>'}):
        '''
        Train the BPE tokenizer from scratch

        Args:
            text(str):The training text
            vocab_size（int):The desired vocab size.
            allow_special(set):A set of special tokens to include.
        '''

        #Preprocess:Replace spaces with 'Ġ'
        #'Ġ'是GPT2 BPE的实现方法
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text =[]
        for i,char in enumerate(text):
            if char == " " and i !=0:
                processed_text.append('Ġ')
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)    # 'IĠHADĠalwaysĠthoughtĠJackĠGisburnĠratherĠaĠcheapĠgenius'
        #使用特殊字符初始化词表
        #start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]  #['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08', '\t', '\n', '\x0b', '\x0c', '\r', '\x0e', '\x0f', '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ...]

        #扩展特殊字符词表和预处理的文本合并
        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)

        #保证'Ġ'在词表中
        if 'Ġ' not in unique_chars:
            unique_chars.append('Ġ')

        #创建字典
        self.vocab = {i:char for i,char in enumerate(unique_chars)}
        self.inverse_vocab = {char:i for i,char in self.vocab.items()}

        #添加特殊字符
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text  into token ids
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        for new_id in range(len(self.vocab),vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None: #没有可以合并的了
                break
            token_ids = self.replace_pair(token_ids,pair_id, new_id)
            self.bpe_merges[pair_id] = new_id
            # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.

        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe').
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # Convert loaded vocabulary to correct format
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        # Handle newline character without adding a new token
        if "\n" not in self.inverse_vocab:
            # Use an existing token ID as a placeholder for '\n'
            # Preferentially use "<|endoftext|>" if available
            fallback_token = next((token for token in ["<|endoftext|>", "Ġ", ""] if token in self.inverse_vocab), None)
            if fallback_token is not None:
                newline_token_id = self.inverse_vocab[fallback_token]
            else:
                # If no fallback token is available, raise an error
                raise KeyError("No suitable token found in vocabulary to map '\\n'.")

            self.inverse_vocab["\n"] = newline_token_id
            self.vocab[newline_token_id] = "\n"

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # Skip header line if present
            if lines and lines[0].startswith("#"):
                lines = lines[1:]

            for line in lines:
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    token1, token2 = pair
                    if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                        token_id1 = self.inverse_vocab[token1]
                        token_id2 = self.inverse_vocab[token2]
                        merged_token = token1 + token2
                        if merged_token in self.inverse_vocab:
                            merged_token_id = self.inverse_vocab[merged_token]
                            self.bpe_merges[(token_id1, token_id2)] = merged_token_id
                        # print(f"Loaded merge: '{token1}' + '{token2}' -> '{merged_token}' (ID: {merged_token_id})")
                        else:
                            print(f"Merged token '{merged_token}' not found in vocab. Skipping.")
                    else:
                        print(f"Skipping pair {pair} as one of the tokens is not in the vocabulary.")

    def encode(self, text):
        """
        Encode the input text into a list of token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: The list of token IDs.
        """
        tokens = []
        # First split on newlines to preserve them
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")  # Add newline token separately
            words = line.split()
            for j, word in enumerate(words):
                if j == 0:
                    if i > 0:  # Start of a new line but not the first line
                        tokens.append("Ġ" + word)  # Ensure it's marked as a new segment
                    else:
                        tokens.append(word)
                else:
                    # Prefix words in the middle of a line with 'Ġ'
                    tokens.append("Ġ" + word)

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                # token is contained in the vocabulary as is
                token_ids.append(self.inverse_vocab[token])
            else:
                # Attempt to handle subword tokenization via BPE
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)

        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.bpe_merges:
                    merged_token_id = self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    # Uncomment for educational purposes:
                    # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                    i += 2  # Skip the next token as it's merged
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return token_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "  # Add space if not present before a newline
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        #统计相邻字符对出现的次数
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced


if __name__ == '__main__':
    import os
    import urllib.request as request

    if not os.path.exists("the-verdict.txt"):
        url = url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
        file_path = "the-verdict.txt"
        request.urlretrieve(url=url,filename=file_path)
    
    with open("the-verdict.txt","r",encoding='utf-8') as f:
        text = f.read()
    print("文本:",text[:50])


    tokenizer = BPETokenizerSimple()
    #预处理文本，扩展预处理文本--添加一些特殊字符   去重-uniquechars
    # 构建字典
    # 文本根据字典转化为tokenids列表  
    # 合并  迭代
    #     查找出最频繁出现的相邻tokenid 对 
    #     将tokenids 中频繁出现的tokenid对合并
    #     将该tokenid对放入一个列表bpe_merge中
    #扩展字典
    #遍历bpe_merge中的每一个对，根据字典取出对应的str,拼接起来加入字典    
    #
    tokenizer.train(text=text,vocab_size=1000,allowed_special={"<|endoftext|>"})
    print(f"vocab_size:{len(tokenizer.vocab)}")
    print(f"vocab:{tokenizer.vocab}")

    print("merge_times:",len(tokenizer.bpe_merges))

    input_text ="Jack embraced beauty through art and life"
    token_ids = tokenizer.encode(input_text)
    print(token_ids)
    print(tokenizer.decode(token_ids))

    for id in token_ids:
        print(f"{id}->{tokenizer.decode([id])}")
        


