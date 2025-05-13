import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from typing import List

from collections import defaultdict
warnings.filterwarnings("ignore")



class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False

class Trie:
    def __init__(self, words, tokenizer):
        self.root = TrieNode()
        self.tokenizer = tokenizer
        for word in words:
            self.insert(word)
    
    def insert(self, word):
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        node = self.root
        for token in tokens:
            node = node.children[token]
        node.is_end = True
    def isPrefix(self, prefix):
        node = self.root
        for token in prefix:
            if token not in node.children:
                return False
            node = node.children[token]
        return True
    def nextTokens(self, prefix_tokens):
        node = self.root
        for token in prefix_tokens:
            if token not in node.children:
                return []
            node = node.children[token]
        return list(node.children.keys())
    def stTokens(self):
        return list(self.root.children.keys())
    

class TrieGen(LogitsProcessor):
    def __init__(self, trie, tokenizer):
        self.trie = trie
        self.tokenizer = tokenizer
        self.boost_factor = 5.6

    def __call__(self, input_ids, scores):
        vocab_size = scores.shape[-1]
        # self.boost_factor = min(2 + 0.5 * len(input_ids[0]), 5) 
  
        seq = input_ids[0,:].tolist()
        lastTok = [seq[-1]]
        if self.trie.isPrefix(lastTok):
            valid_next_tokens = self.trie.nextTokens(lastTok)
            if valid_next_tokens:
                mask = torch.full((vocab_size,), float("-inf"), device=scores.device)
                mask[valid_next_tokens] = 0
                scores[0,:] += mask

        else:
            
            
            first_tokens = self.trie.stTokens()
            scores[0,first_tokens] += self.boost_factor

        return scores


class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
    

        word_list = [word.lower() for word in word_list]
        trie = Trie(word_list,self.tokenizer)

        logits_processor = LogitsProcessorList([TrieGen(trie, self.tokenizer)])


        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + self.max_output_len,
                eos_token_id=self.eos_token_id,
                logits_processor=logits_processor,
                num_beams=5,
                num_return_sequences=1
            )

        # Extract only the generated tokens (excluding input tokens)
        generated_tokens = output[:, input_ids.shape[1]:]
        return generated_tokens.squeeze(0)
        

        raise NotImplementedError
        
        